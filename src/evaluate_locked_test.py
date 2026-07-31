"""Run one guarded, local-only evaluation against the locked YOLO test split.

The evaluator refuses to start without an explicit confirmation flag, requires
a dataset YAML with a test split, and never trains or evaluates train/val data.
It is intentionally a separate workflow from model selection.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import platform
import sys
import time
from pathlib import Path
from typing import Any


DEFAULT_PROJECT = Path("outputs/locked_test")
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}
EXPECTED_TEST_IMAGES = 216
EXPECTED_TEST_LABELS = 216


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--device", default="mps")
    parser.add_argument("--project", type=Path, default=DEFAULT_PROJECT)
    parser.add_argument("--run-name", required=True)
    parser.add_argument(
        "--confirm-locked-test",
        action="store_true",
        help="Explicitly confirm that this is the one-time locked test evaluation.",
    )
    return parser.parse_args(argv)


def repository_root() -> Path:
    return Path(__file__).resolve().parents[1]


def sha256_file(path: Path) -> str:
    if not path.is_file():
        raise FileNotFoundError(f"Cannot hash missing file: {path}")
    digest = hashlib.sha256()
    with path.open("rb") as file_handle:
        for chunk in iter(lambda: file_handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def ensure_confirmation(confirmed: bool) -> None:
    if not confirmed:
        raise PermissionError(
            "Refusing to evaluate: pass --confirm-locked-test to authorise the one-time locked test evaluation."
        )


def parse_dataset_yaml(path: Path) -> dict[str, str]:
    """Parse the small dataset YAML schema without global Ultralytics state."""
    if not path.is_file():
        raise FileNotFoundError(f"Dataset YAML is missing: {path}")
    values: dict[str, str] = {}
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except UnicodeDecodeError as error:
        raise ValueError(f"Dataset YAML is not UTF-8: {path}") from error
    for line in lines:
        stripped = line.split("#", maxsplit=1)[0].strip()
        if not stripped or ":" not in stripped:
            continue
        key, value = (part.strip() for part in stripped.split(":", maxsplit=1))
        if key in {"path", "train", "val", "test", "names"}:
            values[key] = value.strip("\"'")
    if not values.get("test"):
        raise ValueError("Dataset YAML must define a non-empty test split for locked evaluation.")
    if not values.get("path"):
        raise ValueError("Dataset YAML is missing required path.")
    return values


def resolve_dataset_root(root: Path, dataset_yaml: Path) -> tuple[Path, dict[str, str]]:
    values = parse_dataset_yaml(dataset_yaml)
    configured_root = Path(values["path"])
    if configured_root.is_absolute():
        candidates = [configured_root.resolve()]
    else:
        candidates = [
            (dataset_yaml.parent / configured_root).resolve(),
            (root / configured_root).resolve(),
        ]
    unique_candidates = list(dict.fromkeys(candidates))
    valid_candidates = [
        candidate
        for candidate in unique_candidates
        if (candidate / values["test"]).is_dir() and (candidate / "labels" / "test").is_dir()
    ]
    if len(valid_candidates) == 1:
        return valid_candidates[0], values
    if len(valid_candidates) > 1:
        formatted = ", ".join(str(candidate) for candidate in valid_candidates)
        raise ValueError(f"Ambiguous dataset path: multiple candidates contain test images and labels: {formatted}")
    formatted = ", ".join(str(candidate) for candidate in unique_candidates)
    raise FileNotFoundError(
        "Could not resolve dataset root from the YAML path. Tested candidates: "
        f"{formatted}; each must contain {values['test']} and labels/test."
    )


def count_files(directory: Path, suffixes: set[str]) -> int:
    if not directory.is_dir():
        raise FileNotFoundError(f"Test directory is missing: {directory}")
    return sum(path.is_file() and path.suffix.lower() in suffixes for path in directory.iterdir())


def verify_test_pairs(dataset_root: Path, test_value: str) -> dict[str, int]:
    test_images = dataset_root / test_value
    test_labels = dataset_root / "labels" / "test"
    image_paths = sorted(path for path in test_images.iterdir() if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES)
    label_paths = sorted(path for path in test_labels.iterdir() if path.is_file() and path.suffix.lower() == ".txt")
    image_stems = {path.stem for path in image_paths}
    label_stems = {path.stem for path in label_paths}
    missing_labels = sorted(image_stems - label_stems)
    extra_labels = sorted(label_stems - image_stems)
    if missing_labels or extra_labels:
        raise RuntimeError(
            "Test image/label pairing is incomplete: "
            f"missing labels={missing_labels[:5]}, extra labels={extra_labels[:5]}"
        )
    if len(image_paths) != EXPECTED_TEST_IMAGES or len(label_paths) != EXPECTED_TEST_LABELS:
        raise RuntimeError(
            f"Locked test counts must be {EXPECTED_TEST_IMAGES} images and {EXPECTED_TEST_LABELS} labels; "
            f"found {len(image_paths)} images and {len(label_paths)} labels."
        )
    instances = 0
    for label_path in label_paths:
        try:
            instances += sum(1 for line in label_path.read_text(encoding="utf-8").splitlines() if line.strip())
        except UnicodeDecodeError as error:
            raise ValueError(f"Test label is not UTF-8: {label_path}") from error
    return {"test_image_count": len(image_paths), "test_label_count": len(label_paths), "test_instance_count": instances}


def write_runtime_yaml(
    root: Path,
    dataset_root: Path,
    values: dict[str, str],
    timestamp: str | None = None,
) -> Path:
    runtime_timestamp = timestamp or datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    runtime_path = root / "outputs" / "runtime" / f"locked_test_{runtime_timestamp}.yaml"
    if runtime_path.exists():
        raise FileExistsError(f"Runtime YAML already exists; refusing to overwrite: {runtime_path}")
    runtime_path.parent.mkdir(parents=True, exist_ok=True)
    runtime_path.write_text(
        "\n".join(
            (
                "# Generated for one-time locked test evaluation; do not edit.",
                f"path: {dataset_root}",
                "train: images/train",
                "val: images/val",
                f"test: {values['test']}",
                "names:",
                "  0: acne_lesion",
                "",
            )
        ),
        encoding="utf-8",
    )
    return runtime_path


def resolve_project(root: Path, project: Path) -> Path:
    resolved = project.resolve() if project.is_absolute() else (root / project).resolve()
    forbidden = {(root / "data" / "raw").resolve(), (root / "data" / "processed").resolve()}
    if any(resolved == path or path in resolved.parents for path in forbidden):
        raise ValueError("Locked-test outputs may not be written inside data/raw or data/processed.")
    return resolved


def ensure_output_available(project: Path, run_name: str) -> Path:
    if not run_name or Path(run_name).name != run_name or run_name in {".", ".."}:
        raise ValueError("--run-name must be a simple directory name without path separators.")
    run_dir = project / run_name
    marker = run_dir / "LOCKED_TEST_COMPLETE"
    if marker.exists():
        raise FileExistsError(
            f"Locked-test output is already complete: {run_dir}. No overwrite option is available."
        )
    if run_dir.exists():
        raise FileExistsError(
            f"An incomplete locked-test directory already exists: {run_dir}. "
            "Archive it manually, choose a new --run-name, and retry; nothing was overwritten."
        )
    return run_dir


def ensure_mps(device: str) -> None:
    if device.lower() != "mps":
        return
    try:
        import torch
    except ImportError as error:
        raise RuntimeError("PyTorch is not installed; MPS cannot be checked.") from error
    if not torch.backends.mps.is_built() or not torch.backends.mps.is_available():
        raise RuntimeError("MPS was requested but is unavailable. No CPU fallback will be attempted.")


def build_validation_kwargs(runtime_yaml: Path, settings: argparse.Namespace, project: Path) -> dict[str, Any]:
    return {
        "data": str(runtime_yaml),
        "split": "test",
        "imgsz": settings.imgsz,
        "batch": settings.batch,
        "device": settings.device,
        "workers": settings.workers,
        "plots": True,
        "project": str(project),
        "name": settings.run_name,
        "exist_ok": False,
        "verbose": True,
    }


def _metric(results: Any, keys: tuple[str, ...]) -> float | None:
    values = getattr(results, "results_dict", {}) or {}
    for key in keys:
        if key in values:
            return float(values[key])
    box = getattr(results, "box", None)
    for key in keys:
        short = key.rsplit("/", maxsplit=1)[-1].replace("(B)", "")
        candidate = getattr(box, {"metrics/precision": "mp", "metrics/recall": "mr", "metrics/mAP50": "map50", "metrics/mAP50-95": "map"}.get(f"metrics/{short}", ""), None) if box else None
        if candidate is not None:
            return float(candidate)
    return None


def collect_result(results: Any, elapsed_seconds: float, settings: argparse.Namespace, hashes: dict[str, str], counts: dict[str, int], run_dir: Path) -> dict[str, Any]:
    speed = getattr(results, "speed", {}) or {}
    return {
        "test_image_count": counts["test_image_count"],
        "test_label_count": counts["test_label_count"],
        "test_instance_count": counts["test_instance_count"],
        "precision": _metric(results, ("metrics/precision(B)", "metrics/precision")),
        "recall": _metric(results, ("metrics/recall(B)", "metrics/recall")),
        "mAP50": _metric(results, ("metrics/mAP50(B)", "metrics/mAP50")),
        "mAP50-95": _metric(results, ("metrics/mAP50-95(B)", "metrics/mAP50-95")),
        "preprocessing_ms_per_image": speed.get("preprocess"),
        "inference_ms_per_image": speed.get("inference"),
        "postprocessing_ms_per_image": speed.get("postprocess"),
        "total_elapsed_seconds": round(elapsed_seconds, 3),
        "model_sha256": hashes["model_sha256"],
        "dataset_yaml_sha256": hashes["dataset_yaml_sha256"],
        "split_manifest_sha256": hashes["split_manifest_sha256"],
        "runtime_yaml_sha256": hashes.get("runtime_yaml_sha256"),
        "ultralytics_version": __import__("importlib.metadata", fromlist=["version"]).version("ultralytics"),
        "pytorch_version": __import__("torch").__version__,
        "python_version": platform.python_version(),
        "device": settings.device,
        "imgsz": settings.imgsz,
        "batch": settings.batch,
        "workers": settings.workers,
        "split": "test",
        "model": str(settings.model),
        "data": str(settings.data),
        "output_directory": str(run_dir),
    }


def write_outputs(run_dir: Path, report_data: dict[str, Any], runtime_yaml: Path) -> None:
    (run_dir / "LOCKED_TEST_RESULTS.json").write_text(json.dumps(report_data, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    lines = [
        "# Locked ACNE04 Test Evaluation",
        "",
        "This report records one explicitly confirmed test-split evaluation. It is not used for model selection or tuning.",
        "",
        "## Results",
        "",
    ]
    for key in ("test_image_count", "test_instance_count", "precision", "recall", "mAP50", "mAP50-95", "preprocessing_ms_per_image", "inference_ms_per_image", "postprocessing_ms_per_image", "total_elapsed_seconds"):
        lines.append(f"- {key}: {report_data.get(key)}")
    lines.extend(("", "## Provenance", ""))
    for key in ("model_sha256", "dataset_yaml_sha256", "split_manifest_sha256", "runtime_yaml_sha256", "ultralytics_version", "pytorch_version", "python_version", "device", "split", "runtime_yaml"):
        lines.append(f"- {key}: {report_data.get(key, str(runtime_yaml) if key == 'runtime_yaml' else None)}")
    lines.extend(("", "No medical, diagnostic, or production-readiness conclusion is made.", ""))
    (run_dir / "LOCKED_TEST_REPORT.md").write_text("\n".join(lines), encoding="utf-8")
    (run_dir / "LOCKED_TEST_COMPLETE").write_text("One-time locked test evaluation completed.\n", encoding="utf-8")


def evaluate(settings: argparse.Namespace) -> dict[str, Any]:
    ensure_confirmation(settings.confirm_locked_test)
    if settings.imgsz <= 0 or settings.batch <= 0 or settings.workers < 0:
        raise ValueError("imgsz and batch must be positive; workers must be non-negative.")
    root = repository_root()
    dataset_yaml = settings.data.resolve() if settings.data.is_absolute() else (root / settings.data).resolve()
    dataset_root, values = resolve_dataset_root(root, dataset_yaml)
    counts = verify_test_pairs(dataset_root, values["test"])
    manifest = dataset_root / "split_manifest.csv"
    model_path = settings.model.resolve() if settings.model.is_absolute() else (root / settings.model).resolve()
    hashes = {
        "model_sha256": sha256_file(model_path),
        "dataset_yaml_sha256": sha256_file(dataset_yaml),
        "split_manifest_sha256": sha256_file(manifest),
    }
    project = resolve_project(root, settings.project)
    run_dir = ensure_output_available(project, settings.run_name)
    ensure_mps(settings.device)
    runtime_yaml = write_runtime_yaml(root, dataset_root, values)
    hashes["runtime_yaml_sha256"] = sha256_file(runtime_yaml)
    try:
        from ultralytics import YOLO
    except ImportError as error:
        raise RuntimeError("Ultralytics is not installed in the active environment.") from error
    model = YOLO(str(model_path))
    started = time.monotonic()
    results = model.val(**build_validation_kwargs(runtime_yaml, settings, project))
    elapsed = time.monotonic() - started
    if not run_dir.is_dir():
        raise RuntimeError(f"Ultralytics did not create the expected locked-test directory: {run_dir}")
    report_data = collect_result(results, elapsed, settings, hashes, counts, run_dir)
    report_data["runtime_yaml"] = str(runtime_yaml)
    write_outputs(run_dir, report_data, runtime_yaml)
    return report_data


def main() -> None:
    report = evaluate(parse_args())
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
