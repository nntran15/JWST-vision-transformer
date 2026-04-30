import csv
import tempfile
import unittest
from pathlib import Path

import numpy as np
from astropy.io import fits
import torch
import torch.nn as nn

import src.evaluation.classifier as classifier
from src.evaluation.classifier import create_train_val_loaders, train_classifier


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


if __name__ == "__main__":
    unittest.main()