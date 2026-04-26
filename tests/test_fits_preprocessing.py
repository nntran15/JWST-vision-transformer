import numpy as np

from src.data.fits_preprocessing import compose_multiband_rgb


def test_compose_multiband_rgb_suppresses_low_signal_chroma():
    image = np.zeros((4, 5, 5), dtype=np.float32)
    image[0, 2, 2] = 0.08
    image[1, 2, 2] = 0.04
    image[2, 2, 2] = 0.01

    rgb = compose_multiband_rgb(image)

    saturation = float(rgb[2, 2].max() - rgb[2, 2].min())
    assert saturation < 0.03


def test_compose_multiband_rgb_preserves_faint_short_band_luminance():
    image = np.zeros((4, 5, 5), dtype=np.float32)
    image[0, 2, 2] = 0.03
    image[1, 2, 2] = 0.02

    rgb = compose_multiband_rgb(image)
    center = rgb[2, 2]

    assert float(center.mean()) > 0.015
    assert float(center.max() - center.min()) < 0.02


def test_compose_multiband_rgb_preserves_high_signal_color_order():
    image = np.zeros((4, 5, 5), dtype=np.float32)
    image[0, 2, 2] = 0.20
    image[1, 2, 2] = 0.30
    image[2, 2, 2] = 0.60
    image[3, 2, 2] = 0.90

    rgb = compose_multiband_rgb(image)
    center = rgb[2, 2]

    assert center[0] > center[1] > center[2]
    assert center[0] - center[2] > 0.20


def test_compose_multiband_rgb_clips_output_range():
    image = np.full((4, 3, 3), 2.0, dtype=np.float32)

    rgb = compose_multiband_rgb(image)

    assert rgb.shape == (3, 3, 3)
    assert float(rgb.min()) >= 0.0
    assert float(rgb.max()) <= 1.0