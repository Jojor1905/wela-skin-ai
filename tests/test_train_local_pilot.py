"""Unit tests for local-pilot safety and path handling; no model training occurs."""

from __future__ import annotations

import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from train_local_pilot import (  # noqa: E402
    DatasetPaths, ensure_mps, parse_pilot_yaml, repository_root, resolve_dataset_paths,
    unique_run_name, write_runtime_yaml,
)


class LocalPilotTests(unittest.TestCase):
    def test_repository_root_is_repository_parent_of_source(self) -> None:
        self.assertEqual(repository_root().name, "wela-skin-ai")
        self.assertTrue((repository_root() / "src" / "train_local_pilot.py").is_file())

    def test_portable_yaml_resolves_from_repository_root(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            paths = self._make_dataset(root)
            yaml_path = root / "configs" / "pilot.yaml"
            yaml_path.parent.mkdir()
            yaml_path.write_text("path: data/dataset\ntrain: images/train\nval: images/val\nnames:\n  0: acne_lesion\n", encoding="utf-8")
            resolved = resolve_dataset_paths(root, yaml_path)
        self.assertEqual(resolved.dataset_root, paths.dataset_root)
        self.assertEqual((resolved.train_count, resolved.val_count), (1, 1))

    def test_test_split_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            yaml_path = Path(temporary_directory) / "pilot.yaml"
            yaml_path.write_text("path: data\ntrain: images/train\nval: images/val\ntest: images/test\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "must not define test"):
                parse_pilot_yaml(yaml_path)

    def test_runtime_yaml_is_absolute_and_omits_test(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            paths = self._make_dataset(root)
            runtime_yaml = write_runtime_yaml(root, paths)
            contents = runtime_yaml.read_text(encoding="utf-8")
        self.assertIn(f"path: {paths.dataset_root}", contents)
        self.assertNotIn("test:", contents)

    def test_runtime_directory_is_created(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            runtime_yaml = write_runtime_yaml(root, self._make_dataset(root))
            self.assertTrue(runtime_yaml.parent.is_dir())

    def test_missing_train_or_validation_directory_fails(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            (root / "data" / "dataset").mkdir(parents=True)
            yaml_path = root / "pilot.yaml"
            yaml_path.write_text("path: data/dataset\ntrain: images/train\nval: images/val\n", encoding="utf-8")
            with self.assertRaisesRegex(FileNotFoundError, "Train images"):
                resolve_dataset_paths(root, yaml_path)

    def test_missing_validation_directory_fails(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            dataset = root / "data" / "dataset"
            (dataset / "images" / "train").mkdir(parents=True)
            (dataset / "labels" / "train").mkdir(parents=True)
            yaml_path = root / "pilot.yaml"
            yaml_path.write_text("path: data/dataset\ntrain: images/train\nval: images/val\n", encoding="utf-8")
            with self.assertRaisesRegex(FileNotFoundError, "Validation images"):
                resolve_dataset_paths(root, yaml_path)

    def test_image_label_count_mismatch_fails(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            paths = self._make_dataset(root)
            (paths.train_labels / "extra.txt").write_text("", encoding="utf-8")
            yaml_path = root / "pilot.yaml"
            yaml_path.write_text("path: data/dataset\ntrain: images/train\nval: images/val\n", encoding="utf-8")
            with self.assertRaisesRegex(RuntimeError, "Train image/label count mismatch"):
                resolve_dataset_paths(root, yaml_path)

    def test_unique_run_name_avoids_existing_runs(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            project = Path(temporary_directory)
            (project / "pilot").mkdir()
            (project / "pilot_1").mkdir()
            self.assertEqual(unique_run_name(project, "pilot"), "pilot_2")

    def test_mps_unavailable_raises_without_cpu_fallback(self) -> None:
        fake_torch = types.SimpleNamespace(backends=types.SimpleNamespace(mps=types.SimpleNamespace(is_built=lambda: True, is_available=lambda: False)))
        with patch.dict(sys.modules, {"torch": fake_torch}):
            with self.assertRaisesRegex(RuntimeError, "No CPU fallback"):
                ensure_mps("mps")

    def _make_dataset(self, root: Path) -> DatasetPaths:
        dataset = (root / "data" / "dataset").resolve()
        for directory in (dataset / "images" / "train", dataset / "images" / "val", dataset / "labels" / "train", dataset / "labels" / "val"):
            directory.mkdir(parents=True, exist_ok=True)
        (dataset / "images" / "train" / "one.jpg").write_bytes(b"image")
        (dataset / "images" / "val" / "one.jpg").write_bytes(b"image")
        (dataset / "labels" / "train" / "one.txt").write_text("0 0.5 0.5 0.1 0.1\n", encoding="utf-8")
        (dataset / "labels" / "val" / "one.txt").write_text("0 0.5 0.5 0.1 0.1\n", encoding="utf-8")
        return DatasetPaths(dataset, dataset / "images" / "train", dataset / "labels" / "train", dataset / "images" / "val", dataset / "labels" / "val", 1, 1)


if __name__ == "__main__":
    unittest.main()
