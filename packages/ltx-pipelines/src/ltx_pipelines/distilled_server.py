"""Resident distilled inference server (vLLM/SGLang-style, stdlib only).

Loads the two heavy models -- the Gemma text encoder (~23GB) and the transformer
(~18.6GB) -- **once at startup** and keeps them resident for the process
lifetime. Each HTTP request then only pays the compute (both denoising stages)
plus the small per-clip decoder/upsampler builds, instead of re-reading ~40GB
from disk on every call.

Residency is achieved with an ``ExitStack`` that holds open:
  * ``PromptEncoder.text_encoder_context()`` -- resident Gemma
  * ``DiffusionStage.model_context()``        -- resident transformer

Generation is serialised behind a lock (single GPU pipeline -> one clip at a
time). No third-party web framework: uses ``http.server`` from the stdlib.

Run::

    python -m ltx_pipelines.distilled_server \
        --distilled-checkpoint-path .../model-fp8.safetensors \
        --spatial-upsampler-path .../upsampler.safetensors \
        --gemma-root .../gemma \
        --quantization fp8-cast --offload cpu \
        --host 0.0.0.0 --port 8000

Call::

    curl -s -X POST http://localhost:8000/generate \
        -H 'Content-Type: application/json' \
        -d '{"prompt": "a cat playing piano", "seed": 10}' \
        -o out.mp4
"""

import json
import logging
import tempfile
import threading
import time
from contextlib import ExitStack, nullcontext
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import torch

from ltx_core.components.noisers import GaussianNoiser
from ltx_core.loader import LoraPathStrengthAndSDOps
from ltx_core.loader.registry import Registry
from ltx_core.model.transformer.compiling import CompilationConfig
from ltx_core.model.video_vae import TilingConfig, get_video_chunks_number
from ltx_core.quantization import QuantizationPolicy
from ltx_core.types import Audio
from ltx_pipelines.utils.args import default_2_stage_distilled_arg_parser, detect_checkpoint_path
from ltx_pipelines.utils.blocks import (
    AudioDecoder,
    DiffusionStage,
    PromptEncoder,
    VideoDecoder,
    VideoUpsampler,
)
from ltx_pipelines.utils.constants import DISTILLED_SIGMAS, STAGE_2_DISTILLED_SIGMAS, detect_params
from ltx_pipelines.utils.denoisers import SimpleDenoiser
from ltx_pipelines.utils.helpers import assert_resolution, get_device
from ltx_pipelines.utils.media_io import encode_video
from ltx_pipelines.utils.types import ModalitySpec, OffloadMode

logger = logging.getLogger(__name__)


class ResidentDistilledPipeline:
    """Distilled two-stage pipeline holding Gemma + transformer resident."""

    def __init__(
        self,
        distilled_checkpoint_path: str,
        gemma_root: str,
        spatial_upsampler_path: str,
        loras: list[LoraPathStrengthAndSDOps],
        device: torch.device | None = None,
        quantization: QuantizationPolicy | None = None,
        registry: Registry | None = None,
        compilation_config: CompilationConfig | None = None,
        offload_mode: OffloadMode = OffloadMode.NONE,
    ):
        self.device = device or get_device()
        self.dtype = torch.bfloat16

        self.prompt_encoder = PromptEncoder(
            distilled_checkpoint_path, gemma_root, self.dtype, self.device, registry=registry, offload_mode=offload_mode
        )
        self.stage = DiffusionStage(
            distilled_checkpoint_path,
            self.dtype,
            self.device,
            loras=tuple(loras),
            quantization=quantization,
            registry=registry,
            compilation_config=compilation_config,
            offload_mode=offload_mode,
        )
        self.upsampler = VideoUpsampler(
            distilled_checkpoint_path, spatial_upsampler_path, self.dtype, self.device, registry=registry
        )
        self.video_decoder = VideoDecoder(distilled_checkpoint_path, self.dtype, self.device, registry=registry)
        self.audio_decoder = AudioDecoder(distilled_checkpoint_path, self.dtype, self.device, registry=registry)

        self._stack: ExitStack | None = None
        self._text_encoder: torch.nn.Module | None = None
        self._transformer: object | None = None
        self.resident: frozenset[str] = frozenset()
        self.lock = threading.Lock()

    def start(self, resident: frozenset[str] = frozenset({"gemma", "transformer"})) -> None:
        """Build and hold the chosen models resident.

        ``resident`` may contain ``"gemma"`` and/or ``"transformer"``. Anything
        not listed is built (and freed) per request instead.
        """
        self.resident = resident
        logger.info("Loading resident models: %s -- one-time startup cost...", sorted(resident) or "none")
        t0 = time.perf_counter()
        self._stack = ExitStack()
        if "gemma" in resident:
            self._text_encoder = self._stack.enter_context(self.prompt_encoder.text_encoder_context())
        if "transformer" in resident:
            self._transformer = self._stack.enter_context(self.stage.model_context())
        logger.info("Resident models ready in %.1fs. Server is warm.", time.perf_counter() - t0)

    def close(self) -> None:
        if self._stack is not None:
            self._stack.close()
            self._stack = None
            self._text_encoder = None
            self._transformer = None
            self.resident = frozenset()

    @torch.inference_mode()
    def generate(
        self,
        prompt: str,
        seed: int,
        height: int,
        width: int,
        num_frames: int,
        frame_rate: float,
        tiling_config: TilingConfig | None = None,
    ) -> tuple[object, Audio]:
        """Generate a single clip. Returns (video_iterator, audio).

        Whichever of Gemma / transformer was made resident at ``start()`` is
        reused; the rest is built (and freed) for this request.
        """
        if self._stack is None:
            raise RuntimeError("Pipeline not started; call start() first.")
        assert_resolution(height=height, width=width, is_two_stage=True)

        stage_1_sigmas = DISTILLED_SIGMAS.to(dtype=torch.float32, device=self.device)
        stage_2_sigmas = STAGE_2_DISTILLED_SIGMAS.to(dtype=torch.float32, device=self.device)
        stage_1_w, stage_1_h = width // 2, height // 2

        # Encode the prompt: resident Gemma if available, else build it per request.
        if self._text_encoder is not None:
            (ctx_p,) = self.prompt_encoder.encode_with(self._text_encoder, [prompt])
        else:
            (ctx_p,) = self.prompt_encoder([prompt])
        video_context, audio_context = ctx_p.video_encoding, ctx_p.audio_encoding

        noiser = GaussianNoiser(generator=torch.Generator(device=self.device).manual_seed(seed))

        # Use the resident transformer if held, otherwise build one for this request.
        transformer_ctx = (
            nullcontext(self._transformer) if self._transformer is not None else self.stage.model_context()
        )
        with transformer_ctx as transformer:
            # Stage 1: low-resolution generation.
            video_state, audio_state = self.stage.run(
                transformer,
                denoiser=SimpleDenoiser(video_context, audio_context),
                sigmas=stage_1_sigmas,
                noiser=noiser,
                width=stage_1_w,
                height=stage_1_h,
                frames=num_frames,
                fps=frame_rate,
                video=ModalitySpec(context=video_context, conditionings=[]),
                audio=ModalitySpec(context=audio_context),
            )

            # Stage 2: upsample + refine at full resolution.
            upscaled_video_latent = self.upsampler(video_state.latent[:1])
            video_state, audio_state = self.stage.run(
                transformer,
                denoiser=SimpleDenoiser(video_context, audio_context),
                sigmas=stage_2_sigmas,
                noiser=noiser,
                width=width,
                height=height,
                frames=num_frames,
                fps=frame_rate,
                video=ModalitySpec(
                    context=video_context,
                    conditionings=[],
                    noise_scale=stage_2_sigmas[0].item(),
                    initial_latent=upscaled_video_latent,
                ),
                audio=ModalitySpec(
                    context=audio_context,
                    noise_scale=stage_2_sigmas[0].item(),
                    initial_latent=audio_state.latent,
                ),
            )
            video_latent, audio_latent = video_state.latent, audio_state.latent

        # Decode (transformer no longer needed; freed already if it was per-request).
        generator = torch.Generator(device=self.device).manual_seed(seed)
        video = self.video_decoder(video_latent, tiling_config, generator)
        audio = self.audio_decoder(audio_latent)
        return video, audio


def _make_handler(pipeline: ResidentDistilledPipeline, defaults: dict) -> type[BaseHTTPRequestHandler]:
    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, fmt: str, *args: object) -> None:  # quieter default logging
            logger.info("%s - %s", self.address_string(), fmt % args)

        def _send_json(self, code: int, payload: dict) -> None:
            body = json.dumps(payload).encode("utf-8")
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self) -> None:
            if self.path == "/health":
                self._send_json(
                    200,
                    {"status": "ok", "busy": pipeline.lock.locked(), "resident": sorted(pipeline.resident)},
                )
            else:
                self._send_json(404, {"error": "not found"})

        def do_POST(self) -> None:
            if self.path != "/generate":
                self._send_json(404, {"error": "not found"})
                return
            try:
                length = int(self.headers.get("Content-Length", 0))
                req = json.loads(self.rfile.read(length) or b"{}")
            except (ValueError, json.JSONDecodeError) as e:
                self._send_json(400, {"error": f"invalid JSON: {e}"})
                return

            prompt = req.get("prompt")
            if not prompt:
                self._send_json(400, {"error": "missing 'prompt'"})
                return

            params = {
                "seed": int(req.get("seed", defaults["seed"])),
                "height": int(req.get("height", defaults["height"])),
                "width": int(req.get("width", defaults["width"])),
                "num_frames": int(req.get("num_frames", defaults["num_frames"])),
                "frame_rate": float(req.get("frame_rate", defaults["frame_rate"])),
            }

            tiling_config = TilingConfig.default()
            chunks = get_video_chunks_number(params["num_frames"], tiling_config)

            t0 = time.perf_counter()
            # inference_mode must span the lazy video decode too: generate() returns
            # an iterator whose conv ops only run while encode_video consumes it, and
            # this handler runs in a worker thread (inference_mode is thread-local).
            with pipeline.lock, torch.inference_mode():  # one generation at a time
                logger.info("Generating: %r %s", prompt[:60], params)
                try:
                    video, audio = pipeline.generate(prompt=prompt, tiling_config=tiling_config, **params)
                    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp:
                        tmp_path = tmp.name
                    encode_video(
                        video=video,
                        fps=params["frame_rate"],
                        audio=audio,
                        output_path=tmp_path,
                        video_chunks_number=chunks,
                    )
                    data = Path(tmp_path).read_bytes()
                    Path(tmp_path).unlink(missing_ok=True)
                except Exception as e:
                    logger.exception("Generation failed")
                    self._send_json(500, {"error": str(e)})
                    return

            elapsed = time.perf_counter() - t0
            logger.info("Done in %.1fs (%d bytes)", elapsed, len(data))
            self.send_response(200)
            self.send_header("Content-Type", "video/mp4")
            self.send_header("Content-Length", str(len(data)))
            self.send_header("X-Generation-Seconds", f"{elapsed:.1f}")
            self.end_headers()
            self.wfile.write(data)

    return Handler


@torch.inference_mode()
def main() -> None:
    logging.basicConfig(level=logging.INFO)
    checkpoint_path = detect_checkpoint_path(distilled=True)
    params = detect_params(checkpoint_path)
    parser = default_2_stage_distilled_arg_parser(params=params)
    for action in parser._actions:
        if action.dest in {"prompt", "output_path"}:
            action.required = False
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument(
        "--resident",
        choices=["both", "gemma", "transformer", "none"],
        default="both",
        help="Which heavy models to keep resident in memory across requests "
        "(others are built per request). Default: both.",
    )
    args = parser.parse_args()

    resident = {
        "both": frozenset({"gemma", "transformer"}),
        "gemma": frozenset({"gemma"}),
        "transformer": frozenset({"transformer"}),
        "none": frozenset(),
    }[args.resident]

    pipeline = ResidentDistilledPipeline(
        distilled_checkpoint_path=args.distilled_checkpoint_path,
        spatial_upsampler_path=args.spatial_upsampler_path,
        gemma_root=args.gemma_root,
        loras=tuple(args.lora) if args.lora else (),
        quantization=args.quantization,
        compilation_config=args.compile,
        offload_mode=args.offload_mode,
    )
    defaults = {
        "seed": args.seed,
        "height": args.height,
        "width": args.width,
        "num_frames": args.num_frames,
        "frame_rate": args.frame_rate,
    }

    pipeline.start(resident)
    handler = _make_handler(pipeline, defaults)
    server = ThreadingHTTPServer((args.host, args.port), handler)
    logger.info("Serving on http://%s:%d  (POST /generate, GET /health)", args.host, args.port)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        logger.info("Shutting down...")
    finally:
        server.server_close()
        pipeline.close()


if __name__ == "__main__":
    main()
