"""Run pyannote Community-1 locally and emit a stable JSON interchange file."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
from typing import Any, Iterable

from transcriber_diarizer import __version__


DEFAULT_MODEL = "pyannote/speaker-diarization-community-1"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="transcriber-diarize",
        description="Run local pyannote speaker diarization and write JSON.",
    )
    parser.add_argument("--version", action="version", version=__version__)
    parser.add_argument("--audio", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument(
        "--download-only",
        action="store_true",
        help="download and validate the diarization model without processing audio",
    )
    parser.add_argument("--num-speakers", type=positive_int)
    parser.add_argument("--min-speakers", type=positive_int)
    parser.add_argument("--max-speakers", type=positive_int)
    return parser.parse_args(argv)


def positive_int(value: str) -> int:
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError("must be at least 1")
    return number


def annotation_segments(annotation: Any) -> list[dict[str, Any]]:
    if annotation is None:
        return []

    segments: list[dict[str, Any]] = []
    if hasattr(annotation, "itertracks"):
        values: Iterable[Any] = annotation.itertracks(yield_label=True)
        for turn, _, speaker in values:
            segments.append(segment_dict(turn, speaker))
    else:
        for turn, speaker in annotation:
            segments.append(segment_dict(turn, speaker))
    return sorted(segments, key=lambda item: (item["start"], item["end"], item["speaker"]))


def segment_dict(turn: Any, speaker: Any) -> dict[str, Any]:
    return {
        "start": round(float(turn.start), 6),
        "end": round(float(turn.end), 6),
        "speaker": str(speaker),
    }


def select_device(requested: str, torch: Any) -> str:
    if requested == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but PyTorch cannot access a CUDA device")
    return requested


def write_json_atomic(path: Path, document: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(document, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def run(args: argparse.Namespace) -> None:
    if not args.download_only and (args.audio is None or args.output is None):
        raise ValueError("--audio and --output are required unless --download-only is used")
    if args.audio is not None and not args.audio.is_file():
        raise FileNotFoundError(f"audio file not found: {args.audio}")
    if args.num_speakers and (args.min_speakers or args.max_speakers):
        raise ValueError("--num-speakers cannot be combined with a speaker range")
    if args.min_speakers and args.max_speakers and args.min_speakers > args.max_speakers:
        raise ValueError("--min-speakers cannot exceed --max-speakers")

    # Preserve the local-first behavior of the parent application. Model downloads
    # still go through Hugging Face, but inference telemetry is disabled by default.
    os.environ.setdefault("PYANNOTE_METRICS_ENABLED", "0")

    import torch
    from pyannote.audio import Pipeline

    device = select_device(args.device, torch)
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_TOKEN")
    model_path = Path(args.model).expanduser()
    model_source = str(model_path) if model_path.exists() else args.model
    load_options = {"token": token} if token and not model_path.exists() else {}

    print(f"loading diarization model {model_source}", file=sys.stderr)
    pipeline = Pipeline.from_pretrained(model_source, **load_options)
    if pipeline is None:
        raise RuntimeError(
            "the diarization model could not be loaded; accept its Hugging Face "
            "conditions and authenticate with `hf auth login` or a read-scoped HF_TOKEN"
        )
    if args.download_only:
        print(f"diarization model is ready: {model_source}", file=sys.stderr)
        return

    pipeline.to(torch.device(device))

    inference_options: dict[str, int] = {}
    if args.num_speakers:
        inference_options["num_speakers"] = args.num_speakers
    if args.min_speakers:
        inference_options["min_speakers"] = args.min_speakers
    if args.max_speakers:
        inference_options["max_speakers"] = args.max_speakers

    print(f"diarizing on {device}", file=sys.stderr)
    output = pipeline(str(args.audio), **inference_options)
    regular = getattr(output, "speaker_diarization", output)
    exclusive = getattr(output, "exclusive_speaker_diarization", None)
    segments = annotation_segments(regular)
    exclusive_segments = annotation_segments(exclusive)
    speakers = sorted({item["speaker"] for item in segments + exclusive_segments})

    write_json_atomic(
        args.output,
        {
            "model": args.model,
            "device": device,
            "speakers": speakers,
            "segments": segments,
            "exclusive_segments": exclusive_segments,
        },
    )
    print(f"wrote {args.output}", file=sys.stderr)


def main() -> None:
    try:
        run(parse_args())
    except KeyboardInterrupt:
        raise SystemExit(130) from None
    except Exception as error:
        print(f"transcriber-diarize: {error}", file=sys.stderr)
        raise SystemExit(1) from error


if __name__ == "__main__":
    main()
