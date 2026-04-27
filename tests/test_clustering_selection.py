from pathlib import Path
import tempfile
import unittest

import numpy as np
from astropy.io import fits

from src.evaluation.annotation_tool import display_cluster_grid, has_interactive_display
from src.evaluation.clustering import select_best_k


class SelectBestKTests(unittest.TestCase):
    def test_elbow_strategy_avoids_smallest_k_silhouette_bias(self):
        ks = [2, 3, 4, 5, 6, 7, 8, 9, 10]
        inertias = [
            320979.16,
            183590.34,
            129394.03,
            97002.00,
            77580.38,
            67020.93,
            56819.73,
            49581.42,
            44048.48,
        ]
        silhouettes = [0.6954, 0.6189, 0.5374, 0.4479, 0.4335, 0.4099, 0.4203, 0.4040, 0.3776]

        self.assertEqual(select_best_k(ks, inertias, silhouettes, strategy="silhouette"), 2)
        self.assertEqual(select_best_k(ks, inertias, silhouettes, strategy="elbow"), 4)


class AnnotationToolTests(unittest.TestCase):
    def test_headless_display_detection_matches_ci_environment(self):
        self.assertFalse(has_interactive_display())

    def test_display_cluster_grid_saves_preview_when_show_is_disabled(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            fits_path = tmp_path / "source.fits"
            output_path = tmp_path / "cluster_000.png"

            fits.PrimaryHDU(np.ones((16, 16), dtype=np.float32)).writeto(fits_path)

            saved_path = display_cluster_grid(
                cluster_id=0,
                image_paths=[str(fits_path)],
                output_path=str(output_path),
                show=False,
            )

            self.assertEqual(saved_path, output_path)
            self.assertTrue(output_path.exists())


if __name__ == "__main__":
    unittest.main()