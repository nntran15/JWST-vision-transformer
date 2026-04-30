import csv
import tempfile
import unittest
from pathlib import Path

import numpy as np
from astropy.io import fits
import torch
import torch.nn as nn
from torch.utils.data import WeightedRandomSampler

import src.evaluation.classifier as classifier
from src.evaluation.classifier import LinearProbe, create_train_val_loaders, create_train_val_test_loaders, train_classifier


def write_sample_fits(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fits.PrimaryHDU(np.ones((8, 8), dtype=np.float32)).writeto(path)


class CreateTrainValLoadersTests(unittest.TestCase):
    def test_unstratified_fallback_handles_small_dataset(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            csv_path = tmp_path / "labels.csv"

            samples = [
                (tmp_path / "a.fits", "label"),
                (tmp_path / "b.fits", "not_spiral"),
                (tmp_path / "c.fits", "spiral"),
                (tmp_path / "d.fits", "not_spiral"),
            ]
            for fits_path, _ in samples:
                write_sample_fits(fits_path)

            with csv_path.open("w", newline="") as handle:
                writer = csv.writer(handle)
                writer.writerow(["file_path", "cluster_id", "label"])
                for fits_path, label in samples:
                    writer.writerow([str(fits_path), "manual", label])

            train_loader, val_loader, label_to_idx, class_weights = create_train_val_loaders(
                str(csv_path),
                batch_size=1,
                num_workers=0,
                seed=42,
            )

            self.assertEqual(len(label_to_idx), 2)
            self.assertGreaterEqual(len(train_loader.dataset), 1)
            self.assertGreaterEqual(len(val_loader.dataset), 1)
            self.assertEqual(class_weights.shape[0], 2)

    def test_create_train_val_test_loaders_uses_frozen_split_artifact(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            csv_path = tmp_path / "labels.csv"
            split_dir = tmp_path / "split"
            split_dir.mkdir()

            samples = [
                (tmp_path / "a.fits", "c0", "not_spiral"),
                (tmp_path / "b.fits", "c1", "spiral"),
                (tmp_path / "c.fits", "c2", "not_spiral"),
                (tmp_path / "d.fits", "c3", "spiral"),
                (tmp_path / "e.fits", "c4", "not_spiral"),
                (tmp_path / "f.fits", "c5", "spiral"),
            ]
            for fits_path, _, _ in samples:
                write_sample_fits(fits_path)

            with csv_path.open("w", newline="") as handle:
                writer = csv.writer(handle)
                writer.writerow(["file_path", "cluster_id", "label"])
                for fits_path, cluster_id, label in samples:
                    writer.writerow([str(fits_path), cluster_id, label])

            def write_split_csv(path: Path, rows: list[tuple[Path, str, str]]) -> None:
                with path.open("w", newline="") as handle:
                    writer = csv.writer(handle)
                    writer.writerow(["file_path", "cluster_id", "label"])
                    for fits_path, cluster_id, label in rows:
                        writer.writerow([str(fits_path), cluster_id, label])

            write_split_csv(split_dir / "train.csv", [samples[0], samples[1]])
            write_split_csv(split_dir / "val.csv", [samples[2], samples[3]])
            write_split_csv(split_dir / "test.csv", [samples[4], samples[5]])
            (split_dir / "split_manifest.json").write_text("{}")

            train_loader, val_loader, test_loader, label_to_idx, class_weights, split_metadata = create_train_val_test_loaders(
                str(csv_path),
                batch_size=1,
                num_workers=0,
                split_artifact=str(split_dir),
                seed=42,
            )

            self.assertEqual(len(label_to_idx), 2)
            self.assertEqual(len(train_loader.dataset), 2)
            self.assertEqual(len(val_loader.dataset), 2)
            self.assertEqual(len(test_loader.dataset), 2)
            self.assertEqual(class_weights.shape[0], 2)
            self.assertEqual(split_metadata["split_source"], "artifact")
            self.assertEqual(split_metadata["train_count"], 2)
            self.assertEqual(split_metadata["val_count"], 2)
            self.assertEqual(split_metadata["test_count"], 2)

    def test_balanced_sampling_disables_loss_class_weights(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            csv_path = tmp_path / "labels.csv"

            samples = [
                (tmp_path / "n0.fits", "c0", "not_spiral"),
                (tmp_path / "n1.fits", "c1", "not_spiral"),
                (tmp_path / "n2.fits", "c2", "not_spiral"),
                (tmp_path / "n3.fits", "c3", "not_spiral"),
                (tmp_path / "s0.fits", "c4", "spiral"),
                (tmp_path / "s1.fits", "c5", "spiral"),
            ]
            for fits_path, _, _ in samples:
                write_sample_fits(fits_path)

            with csv_path.open("w", newline="") as handle:
                writer = csv.writer(handle)
                writer.writerow(["file_path", "cluster_id", "label"])
                for fits_path, cluster_id, label in samples:
                    writer.writerow([str(fits_path), cluster_id, label])

            train_loader, val_loader, test_loader, label_to_idx, class_weights, split_metadata = create_train_val_test_loaders(
                str(csv_path),
                batch_size=2,
                num_workers=0,
                seed=42,
                balanced_sampling=True,
            )

            self.assertEqual(len(label_to_idx), 2)
            self.assertIsNone(class_weights)
            self.assertIsInstance(train_loader.sampler, WeightedRandomSampler)
            self.assertGreater(len(val_loader.dataset), 0)
            self.assertGreater(len(test_loader.dataset), 0)
            self.assertIn("train_class_counts", split_metadata)

    def test_ignores_literal_header_label_row(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            csv_path = tmp_path / "labels.csv"

            samples = [
                (tmp_path / "a.fits", "label"),
                (tmp_path / "b.fits", "not_spiral"),
                (tmp_path / "c.fits", "spiral"),
            ]
            for fits_path, _ in samples:
                write_sample_fits(fits_path)

            with csv_path.open("w", newline="") as handle:
                writer = csv.writer(handle)
                writer.writerow(["file_path", "cluster_id", "label"])
                for fits_path, label in samples:
                    writer.writerow([str(fits_path), "manual", label])

            dataset = classifier.LabeledFITSDataset(str(csv_path))

            self.assertEqual(dataset.n_classes, 2)
            self.assertNotIn("label", dataset.label_to_idx)

    def test_remaps_stale_absolute_paths_to_current_catalog(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            catalog_dir = tmp_path / "data" / "JWST" / "resized_10k_files"
            split_dir = tmp_path / "data" / "JWST" / "splits" / "manual_spiral_v1"
            csv_path = split_dir / "labels_snapshot.csv"

            actual_fits = catalog_dir / "galaxy_a.fits"
            actual_fits.parent.mkdir(parents=True, exist_ok=True)
            fits.PrimaryHDU(np.arange(64, dtype=np.float32).reshape(8, 8)).writeto(actual_fits)

            split_dir.mkdir(parents=True, exist_ok=True)
            with csv_path.open("w", newline="") as handle:
                writer = csv.writer(handle)
                writer.writerow(["file_path", "cluster_id", "label"])
                writer.writerow([
                    "/stale/workstation/data/JWST/resized_10k_files/galaxy_a.fits",
                    "manual",
                    "spiral",
                ])

            dataset = classifier.LabeledFITSDataset(str(csv_path))
            tensor, label_idx = dataset[0]

            self.assertEqual(dataset.resolved_paths[0], str(actual_fits))
            self.assertGreater(float(tensor.sum().item()), 0.0)
            self.assertEqual(label_idx, dataset.label_to_idx["spiral"])

    def test_train_classifier_final_report_handles_missing_class(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            output_dir = tmp_path / "out"

            class DummyModel(nn.Module):
                def __init__(self):
                    super().__init__()
                    self.head = nn.Linear(1, 3)

                def forward(self, x):
                    return self.head(x)

            model = DummyModel()

            train_loader = object()
            val_loader = type(
                "DummyValLoader",
                (),
                {"dataset": type("DummyDataset", (), {"idx_to_label": {0: "not_spiral", 1: "spiral", 2: "artifact"}})()},
            )()

            def fake_evaluate(*args, **kwargs):
                return 0.0, 1.0, np.array([0, 1]), np.array([0, 1])

            original_evaluate = classifier.evaluate
            try:
                classifier.evaluate = fake_evaluate
                result = train_classifier(
                    model=model,
                    train_loader=train_loader,
                    val_loader=val_loader,
                    epochs=0,
                    lr=1e-3,
                    device="cpu",
                    output_dir=str(output_dir),
                )
            finally:
                classifier.evaluate = original_evaluate

            self.assertIn("classification_report", result)
            self.assertTrue((output_dir / "classification_report.txt").exists())

    def test_linear_probe_keeps_encoder_in_eval_mode(self):
        class DummyEncoder(nn.Module):
            def __init__(self):
                super().__init__()
                self.dropout = nn.Dropout(p=0.5)

            def get_cls_token(self, x):
                return self.dropout(x)

        probe = LinearProbe(DummyEncoder(), embed_dim=4, n_classes=2)
        probe.train()

        self.assertFalse(probe.encoder.training)
        self.assertTrue(probe.head.training)


if __name__ == "__main__":
    unittest.main()