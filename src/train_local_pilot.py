"""Run a guarded local-only ACNE04 YOLO pilot on Apple Silicon MPS.

The script uses only train and validation data, writes an absolute-path runtime
YAML, and never changes source or processed dataset files.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any


DEFAULT_PROJECT = Path("outputs/experiments")
RUNTIME_YAML = Path("outputs/runtime/local_pilot.yaml")
READINESS_REPORT = Path("outputs/readiness/READINESS_REPORT.md")
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}


@dataclass(frozen=True)
class DatasetPaths:
    """Validated local pilot directories and their paired file counts."""

    dataset_root: Path
    train_images: Path
    train_labels: Path
    val_images: Path
    val_labels: Path
    train_count: int
    val_count: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("configs/local_pilot.yaml"))
    parser.add_argument("--model", default="yolo26n.pt")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=2)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--device", default="mps")
    parser.add_argument("--project", type=Path, default=DEFAULT_PROJECT)
    parser.add_argument("--run-name", default="local_pilot_smoke")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument(
        "--cache",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Cache images for training (disabled by default).",
    )
    parser.add_argument(
        "--pretrained",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use the supplied model weights as pretrained weights (default: enabled).",
    )
    return parser.parse_args()


def repository_root() -> Path:
    """Locate the repository from this source file, not the working directory."""
    return Path(__file__).resolve().parents[1]


def ensure_python_311() -> None:
    if sys.version_info[:2] != (3, 11):
        raise RuntimeError(f"Python 3.11 is required; found {sys.version_info.major}.{sys.version_info.minor}.")


def ensure_readiness_report(root: Path) -> None:
    report = root / READINESS_REPORT
    if not report.is_file():
        raise FileNotFoundError(f"Readiness report is missing: {report}")
    if "READY FOR PILOT TRAINING" not in report.read_text(encoding="utf-8"):
        raise RuntimeError("The readiness report does not approve local pilot training.")


def parse_pilot_yaml(path: Path) -> dict[str, str]:
    """Parse the small, portable YAML schema without using global Ultralytics state."""
    if not path.is_file():
        raise FileNotFoundError(f"Pilot YAML is missing: {path}")
    values: dict[str, str] = {}
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except UnicodeDecodeError as error:
        raise ValueError(f"Pilot YAML is not UTF-8: {path}") from error
    for line in lines:
        stripped = line.split("#", maxsplit=1)[0].strip()
        if not stripped or ":" not in stripped:
            continue
        key, value = (part.strip() for part in stripped.split(":", maxsplit=1))
        if key in {"path", "train", "val", "test"}:
            values[key] = value.strip("\"'")
    if values.get("test"):
        raise ValueError("Pilot YAML must not define test; the test split is excluded from pilot decisions.")
    missing = [key for key in ("path", "train", "val") if not values.get(key)]
    if missing:
        raise ValueError(f"Pilot YAML is missing required key(s): {', '.join(missing)}")
    return values


def count_files(directory: Path, suffixes: set[str]) -> int:
    return sum(path.is_file() and path.suffix.lower() in suffixes for path in directory.iterdir())


def resolve_dataset_paths(root: Path, pilot_yaml: Path) -> DatasetPaths:
    """Resolve relative data paths from the repository root, never YAML location."""
    values = parse_pilot_yaml(pilot_yaml)
    configured_root = Path(values["path"])
    dataset_root = configured_root.resolve() if configured_root.is_absolute() else (root / configured_root).resolve()
    train_images = dataset_root / values["train"]
    val_images = dataset_root / values["val"]
    train_labels = dataset_root / "labels" / "train"
    val_labels = dataset_root / "labels" / "val"
    for description, directory in (
        ("Dataset root", dataset_root), ("Train images", train_images), ("Train labels", train_labels),
        ("Validation images", val_images), ("Validation labels", val_labels),
    ):
        if not directory.is_dir():
            raise FileNotFoundError(f"{description} directory is missing: {directory}")
    train_count = count_files(train_images, IMAGE_SUFFIXES)
    val_count = count_files(val_images, IMAGE_SUFFIXES)
    train_label_count = count_files(train_labels, {".txt"})
    val_label_count = count_files(val_labels, {".txt"})
    if train_count != train_label_count:
        raise RuntimeError(f"Train image/label count mismatch: {train_count} images, {train_label_count} labels.")
    if val_count != val_label_count:
        raise RuntimeError(f"Validation image/label count mismatch: {val_count} images, {val_label_count} labels.")
    return DatasetPaths(dataset_root, train_images, train_labels, val_images, val_labels, train_count, val_count)


def write_runtime_yaml(root: Path, paths: DatasetPaths) -> Path:
    """Create the only YAML passed to Ultralytics, with an absolute local root."""
    runtime_path = root / RUNTIME_YAML
    runtime_path.parent.mkdir(parents=True, exist_ok=True)
    runtime_path.write_text(
        "\n".join((
            "# Generated local runtime configuration; do not edit source dataset YAML.",
            f"path: {paths.dataset_root}",
            "train: images/train",
            "val: images/val",
            "names:",
            "  0: acne_lesion",
            "",
        )),
        encoding="utf-8",
    )
    return runtime_path


def ensure_mps(device: str) -> None:
    if device.lower() != "mps":
        return
    try:
        import torch
    except ImportError as error:
        raise RuntimeError("PyTorch is not installed; MPS cannot be checked.") from error
    if not torch.backends.mps.is_built() or not torch.backends.mps.is_available():
        raise RuntimeError(
            "MPS was requested but is unavailable. Confirm that this is an Apple Silicon Python 3.11 environment "
            "with a usable MPS device. No CPU fallback will be attempted."
        )


def resolve_project(root: Path, project: Path) -> Path:
    allowed = (root / DEFAULT_PROJECT).resolve()
    resolved = project.resolve() if project.is_absolute() else (root / project).resolve()
    if resolved != allowed and allowed not in resolved.parents:
        raise ValueError(f"--project must be {allowed} or a subdirectory; received {resolved}")
    return resolved


def unique_run_name(project: Path, requested: str) -> str:
    """Avoid overwrite: deterministically add a numeric suffix when needed."""
    if not (project / requested).exists():
        return requested
    index = 1
    while (project / f"{requested}_{index}").exists():
        index += 1
    return f"{requested}_{index}"


def resolve_model(root: Path, model: str) -> Path:
    """Require a local pretrained weight file; never trigger an implicit download."""
    supplied = Path(model)
    candidates = [supplied, root / supplied]
    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()
    raise FileNotFoundError(
        f"Pretrained model weights are not available locally: {model}. Ultralytics may need network access to download "
        "yolo26n.pt; request approval before downloading and do not substitute another model automatically."
    )


def print_dataset_preflight(paths: DatasetPaths, runtime_yaml: Path) -> None:
    print(f"Runtime YAML: {runtime_yaml}")
    print(f"Train images: {paths.train_images} ({paths.train_count})")
    print(f"Train labels: {paths.train_labels} ({paths.train_count})")
    print(f"Validation images: {paths.val_images} ({paths.val_count})")
    print(f"Validation labels: {paths.val_labels} ({paths.val_count})")


def metric_value(row: dict[str, str], keys: tuple[str, ...]) -> float | None:
    for key in keys:
        try:
            return float(row.get(key, ""))
        except ValueError:
            continue
    return None


def write_summary(run_dir: Path, settings: argparse.Namespace, elapsed_seconds: float) -> dict[str, Any]:
    results_path = run_dir / "results.csv"
    if not results_path.is_file():
        raise FileNotFoundError(f"Training output is incomplete: results.csv is missing from {run_dir}")
    with results_path.open(newline="", encoding="utf-8") as file_handle:
        rows = list(csv.DictReader(file_handle))
    if not rows:
        raise RuntimeError("Training output is incomplete: results.csv has no epoch records.")
    best_row = max(rows, key=lambda row: metric_value(row, ("metrics/mAP50-95(B)", "metrics/mAP50-95")) or float("-inf"))
    summary = {
        "data": str(settings.data),
        "model": settings.model,
        "epochs": settings.epochs,
        "image_size": settings.imgsz,
        "batch_size": settings.batch,
        "workers": settings.workers,
        "device": settings.device,
        "project": str(settings.project),
        "run_name": settings.run_name,
        "seed": settings.seed,
        "patience": settings.patience,
        "cache": settings.cache,
        "pretrained": settings.pretrained,
        "elapsed_time_seconds": round(elapsed_seconds, 3),
        "best_validation_precision": metric_value(best_row, ("metrics/precision(B)", "metrics/precision")),
        "best_validation_recall": metric_value(best_row, ("metrics/recall(B)", "metrics/recall")),
        "best_validation_mAP50": metric_value(best_row, ("metrics/mAP50(B)", "metrics/mAP50")),
        "best_validation_mAP50_95": metric_value(best_row, ("metrics/mAP50-95(B)", "metrics/mAP50-95")),
        "output_directory": str(run_dir),
        "actual_run_name": run_dir.name,
    }
    (run_dir / "local_pilot_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return summary


def train(settings: argparse.Namespace) -> dict[str, Any]:
    """Fine-tune local pretrained weights on train/validation data only."""
    root = repository_root()
    ensure_python_311()
    ensure_readiness_report(root)
    pilot_yaml = settings.data.resolve() if settings.data.is_absolute() else (root / settings.data).resolve()
    paths = resolve_dataset_paths(root, pilot_yaml)
    runtime_yaml = write_runtime_yaml(root, paths)
    print_dataset_preflight(paths, runtime_yaml)
    if (
        settings.epochs <= 0
        or settings.imgsz <= 0
        or settings.batch <= 0
        or settings.workers < 0
        or settings.patience < 0
    ):
        raise ValueError("epochs, imgsz, and batch must be positive; workers and patience must be non-negative.")
    ensure_mps(settings.device)
    model_path = resolve_model(root, settings.model)
    project = resolve_project(root, settings.project)
    project.mkdir(parents=True, exist_ok=True)
    run_name = unique_run_name(project, settings.run_name)
    try:
        from ultralytics import YOLO
    except ImportError as error:
        raise RuntimeError("Ultralytics is not installed in the active environment.") from error
    try:
        model = YOLO(str(model_path))
        started = time.monotonic()
        model.train(
            data=str(runtime_yaml), epochs=settings.epochs, imgsz=settings.imgsz, batch=settings.batch,
            workers=settings.workers, device=settings.device, project=str(project), name=run_name,
            seed=settings.seed, patience=settings.patience, cache=settings.cache,
            pretrained=settings.pretrained, val=True, deterministic=True,
        )
        elapsed = time.monotonic() - started
    except RuntimeError as error:
        message = str(error).lower()
        if "out of memory" in message or ("mps" in message and "memory" in message):
            raise RuntimeError("MPS memory exhausted. Reduce --batch or --imgsz; no CPU fallback was used.") from error
        if "mps" in message and ("not implemented" in message or "unsupported" in message):
            raise RuntimeError(
                "An unsupported MPS operation occurred. Do not enable PYTORCH_ENABLE_MPS_FALLBACK automatically; "
                "capture this error and decide explicitly whether a supported configuration is available."
            ) from error
        raise
    run_dir = Path(model.trainer.save_dir)
    for required in (run_dir / "results.csv", run_dir / "weights" / "best.pt", run_dir / "weights" / "last.pt"):
        if not required.is_file():
            raise RuntimeError(f"Training output is incomplete: required artifact is missing: {required}")
    summary = write_summary(run_dir, settings, elapsed)
    print(f"Actual run directory: {run_dir}")
    return summary


def main() -> None:
    summary = train(parse_args())
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
