"""Unit tests for the locked test evaluator; no model is loaded or executed."""

from __future__ import annotations

import hashlib
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from evaluate_locked_test import (  # noqa: E402
    build_validation_kwargs,
    ensure_confirmation,
    ensure_mps,
    ensure_output_available,
    parse_args,
    parse_dataset_yaml,
    resolve_dataset_root,
    sha256_file,
    verify_test_pairs,
    write_runtime_yaml,
)


class LockedTestEvaluationTests(unittest.TestCase):
    def test_confirmation_is_required(self) -> None:
        with self.assertRaisesRegex(PermissionError, "confirm-locked-test"):
            ensure_confirmation(False)
        ensure_confirmation(True)

    def test_missing_test_split_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            path = Path(temporary_directory) / "data.yaml"
            path.write_text("path: data\ntrain: images/train\nval: images/val\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "test split"):
                parse_dataset_yaml(path)

    def test_runtime_yaml_resolves_absolute_dataset_root(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            dataset = root / "data" / "dataset"
            self._make_test_directories(dataset)
            yaml_path = root / "dataset.yaml"
            yaml_path.write_text("path: data/dataset\ntest: images/test\nnames:\n  0: acne_lesion\n", encoding="utf-8")
            dataset_root, values = resolve_dataset_root(root, yaml_path)
            runtime = write_runtime_yaml(root, dataset_root, values, "locked")
            contents = runtime.read_text(encoding="utf-8")
        self.assertIn(f"path: {dataset.resolve()}", contents)
        self.assertIn("train: images/train", contents)
        self.assertIn("val: images/val", contents)
        self.assertIn("test: images/test", contents)

    def test_path_dot_resolves_relative_to_yaml_parent(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory) / "repo"
            dataset = root / "data" / "dataset"
            self._make_test_directories(dataset)
            yaml_path = dataset / "acne04.yaml"
            yaml_path.write_text("path: .\ntest: images/test\n", encoding="utf-8")
            resolved, _ = resolve_dataset_root(root, yaml_path)
        self.assertEqual(resolved, dataset.resolve())

    def test_project_relative_path_resolves_from_repository_root(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory) / "repo"
            dataset = root / "data" / "dataset"
            self._make_test_directories(dataset)
            yaml_path = root / "configs" / "acne04.yaml"
            yaml_path.parent.mkdir(parents=True)
            yaml_path.write_text("path: data/dataset\ntest: images/test\n", encoding="utf-8")
            resolved, _ = resolve_dataset_root(root, yaml_path)
        self.assertEqual(resolved, dataset.resolve())

    def test_absolute_dataset_path_is_used_directly(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory) / "repo"
            dataset = Path(temporary_directory) / "external-dataset"
            self._make_test_directories(dataset)
            yaml_path = root / "configs" / "acne04.yaml"
            yaml_path.parent.mkdir(parents=True)
            yaml_path.write_text(f"path: {dataset}\ntest: images/test\n", encoding="utf-8")
            resolved, _ = resolve_dataset_root(root, yaml_path)
        self.assertEqual(resolved, dataset.resolve())

    def test_invalid_dataset_path_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory) / "repo"
            root.mkdir(parents=True)
            yaml_path = root / "acne04.yaml"
            yaml_path.write_text("path: missing\ntest: images/test\n", encoding="utf-8")
            with self.assertRaisesRegex(FileNotFoundError, "Could not resolve dataset root"):
                resolve_dataset_root(root, yaml_path)

    def test_ambiguous_valid_dataset_paths_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory) / "repo"
            yaml_path = root / "configs" / "acne04.yaml"
            yaml_path.parent.mkdir(parents=True)
            self._make_test_directories(root / "data" / "dataset")
            self._make_test_directories(root / "configs" / "data" / "dataset")
            yaml_path.write_text("path: data/dataset\ntest: images/test\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "Ambiguous dataset path"):
                resolve_dataset_root(root, yaml_path)

    def test_existing_completion_marker_refuses_overwrite(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            project = Path(temporary_directory)
            run_dir = project / "locked"
            run_dir.mkdir()
            (run_dir / "LOCKED_TEST_COMPLETE").write_text("done\n", encoding="utf-8")
            with self.assertRaisesRegex(FileExistsError, "No overwrite"):
                ensure_output_available(project, "locked")

    def test_no_overwrite_option_is_exposed(self) -> None:
        with self.assertRaises(SystemExit):
            parse_args(["--model", "model.pt", "--data", "data.yaml", "--run-name", "run", "--overwrite"])

    def test_mps_unavailable_raises(self) -> None:
        fake_torch = types.SimpleNamespace(
            backends=types.SimpleNamespace(
                mps=types.SimpleNamespace(is_built=lambda: True, is_available=lambda: False)
            )
        )
        with patch.dict(sys.modules, {"torch": fake_torch}):
            with self.assertRaisesRegex(RuntimeError, "No CPU fallback"):
                ensure_mps("mps")

    def test_missing_image_label_pair_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            dataset = Path(temporary_directory)
            images = dataset / "images" / "test"
            labels = dataset / "labels" / "test"
            images.mkdir(parents=True)
            labels.mkdir(parents=True)
            for index in range(216):
                (images / f"image_{index}.jpg").write_bytes(b"x")
                (labels / f"image_{index}.txt").write_text("0 0.5 0.5 0.1 0.1\n", encoding="utf-8")
            (labels / "image_215.txt").unlink()
            with self.assertRaisesRegex(RuntimeError, "pairing is incomplete"):
                verify_test_pairs(dataset, "images/test")

    def test_expected_216_image_label_pairs_are_verified(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            dataset = Path(temporary_directory)
            self._make_test_directories(dataset, pairs=216)
            counts = verify_test_pairs(dataset, "images/test")
        self.assertEqual(counts["test_image_count"], 216)
        self.assertEqual(counts["test_label_count"], 216)

    def test_preflight_check_does_not_create_run_directory(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            project = Path(temporary_directory)
            run_dir = ensure_output_available(project, "new_locked_run")
            self.assertFalse(run_dir.exists())

    def test_incomplete_directory_requires_manual_archiving(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            project = Path(temporary_directory)
            (project / "locked").mkdir()
            with self.assertRaisesRegex(FileExistsError, "Archive it manually"):
                ensure_output_available(project, "locked")

    def test_model_hash_generation(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            path = Path(temporary_directory) / "best.pt"
            path.write_bytes(b"locked-model")
            expected = hashlib.sha256(b"locked-model").hexdigest()
            self.assertEqual(sha256_file(path), expected)

    def test_validation_split_is_always_test(self) -> None:
        settings = types.SimpleNamespace(imgsz=640, batch=1, device="mps", workers=0, run_name="locked")
        kwargs = build_validation_kwargs(Path("runtime.yaml"), settings, Path("outputs/locked_test"))
        self.assertEqual(kwargs["split"], "test")
        self.assertEqual(kwargs["imgsz"], 640)
        self.assertEqual(kwargs["batch"], 1)
        self.assertFalse(kwargs["exist_ok"])

    @staticmethod
    def _make_test_directories(dataset: Path, pairs: int = 0) -> None:
        (dataset / "images" / "test").mkdir(parents=True, exist_ok=True)
        (dataset / "labels" / "test").mkdir(parents=True, exist_ok=True)
        for index in range(pairs):
            (dataset / "images" / "test" / f"image_{index}.jpg").write_bytes(b"x")
            (dataset / "labels" / "test" / f"image_{index}.txt").write_text(
                "0 0.5 0.5 0.1 0.1\n", encoding="utf-8"
            )


if __name__ == "__main__":
    unittest.main()
