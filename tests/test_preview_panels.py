import numpy as np

from src.data.preview_panels import build_source_qa_panel, normalize_stack_with_n_sigma


FILTERS = ("f115w", "f150w", "f277w", "f444w")


def build_raw_crops() -> dict[str, np.ndarray]:
    base = np.zeros((24, 24), dtype=np.float32)
    base[8:16, 8:16] = 1.0
    return {
        "f115w": base * 0.8,
        "f150w": base * 1.0,
        "f277w": base * 1.4,
        "f444w": base * 1.8,
    }


def build_local_rms() -> dict[str, float]:
    return {filt: 0.05 for filt in FILTERS}


def test_normalize_stack_with_n_sigma_changes_stretch():
    raw_crops = build_raw_crops()
    local_rms = build_local_rms()

    low_sigma = normalize_stack_with_n_sigma(raw_crops, local_rms, FILTERS, 4.0)
    high_sigma = normalize_stack_with_n_sigma(raw_crops, local_rms, FILTERS, 20.0)

    assert low_sigma.shape == high_sigma.shape == (4, 24, 24)
    assert float(low_sigma.mean()) > float(high_sigma.mean())


def test_build_source_qa_panel_returns_rgb_panel():
    raw_crops = build_raw_crops()
    local_rms = build_local_rms()
    normalized_stack = normalize_stack_with_n_sigma(raw_crops, local_rms, FILTERS, 8.0)

    panel, rgb = build_source_qa_panel(
        source_label="COSMOS_0000001_A1 | SNR=42.0",
        normalized_stack=normalized_stack,
        raw_crops=raw_crops,
        local_rms_dict=local_rms,
        filters=FILTERS,
        default_n_sigma=8.0,
        brightness_sweep=(1.0,),
        contrast_sweep=(1.0,),
        n_sigma_sweep=(4.0, 8.0),
    )

    assert panel.mode == "RGB"
    assert panel.size[0] > 0
    assert panel.size[1] > 0
    assert rgb.shape == (24, 24, 3)