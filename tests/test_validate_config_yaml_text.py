import copy

import yaml

from tile_compile_backend.validate import validate_config_yaml_text


def _base_cfg() -> dict:
    return {
        "pipeline": {"mode": "production", "abort_on_fail": True},
        "data": {
            "image_width": 3840,
            "image_height": 2160,
            "frames_min": 10,
            "frames_target": 1000,
            "color_mode": "OSC",
            "bayer_pattern": "GBRG",
            "linear_required": True,
        },
        "normalization": {"enabled": True, "mode": "background", "per_channel": True},
        "registration": {
            "engine": "opencv_cfa",
            "reference": "auto",
            "output_dir": "registered",
            "registered_filename_pattern": "reg_{index:05d}.fit",
            "min_star_matches": 10,
            "allow_rotation": False,
        },
        "global_metrics": {
            "weights": {"background": 0.4, "noise": 0.3, "gradient": 0.3},
            "clamp": [-3, 3],
        },
        "tile": {
            "size_factor": 32,
            "min_size": 64,
            "max_divisor": 6,
            "overlap_fraction": 0.25,
            "star_min_count": 3,
        },
        "local_metrics": {
            "clamp": [-3, 3],
            "star_mode": {
                "fwhm_transform": "log",
                "weights": {"fwhm": 0.6, "roundness": 0.2, "contrast": 0.2},
            },
            "structure_mode": {"metric": "energy_over_sigma", "background_weight": 0.3, "metric_weight": 0.7},
        },
        "synthetic": {
            "frames_min": 15,
            "frames_max": 30,
            "clustering": {
                "mode": "state_vector",
                "k_selection": "silhouette_auto",
                "cluster_count_range": [15, 30],
                "vector": ["global_weight", "tile_quality_mean", "tile_quality_variance", "background", "noise"],
            },
        },
        "reconstruction": {
            "weighting_function": "exponential",
            "window_function": "hanning",
            "tile_rescale": "median_after_background_subtraction",
        },
        "stacking": {
            "engine": "siril",
            "method": "average",
            "input_dir": "synthetic",
            "input_pattern": "syn_*.fits",
            "output_file": "stacked.fit",
        },
        "validation": {
            "min_fwhm_improvement_percent": 5,
            "max_background_rms_increase_percent": 0,
            "min_tile_weight_variance": 0.1,
            "require_no_tile_pattern": True,
        },
        "runtime_limits": {"tile_analysis_max_factor_vs_stack": 3.0, "hard_abort_hours": 6},
    }


def _validate(cfg: dict) -> dict:
    yaml_text = yaml.safe_dump(cfg, sort_keys=False)
    return validate_config_yaml_text(yaml_text=yaml_text)


def _codes(result: dict) -> set[str]:
    return {e.get("code") for e in result.get("errors", [])}


def test_valid_config_is_valid() -> None:
    res = _validate(_base_cfg())
    assert res["valid"] is True
    assert res["errors"] == []


def test_global_weights_not_normalized() -> None:
    cfg = _base_cfg()
    cfg["global_metrics"]["weights"]["gradient"] = 0.4
    res = _validate(cfg)
    assert res["valid"] is False
    assert "global_weights_not_normalized" in _codes(res)


def test_local_star_weights_not_normalized() -> None:
    cfg = _base_cfg()
    cfg["local_metrics"]["star_mode"]["weights"]["contrast"] = 0.3
    res = _validate(cfg)
    assert res["valid"] is False
    assert "local_star_weights_not_normalized" in _codes(res)


def test_local_structure_weights_not_normalized() -> None:
    cfg = _base_cfg()
    cfg["local_metrics"]["structure_mode"]["metric_weight"] = 0.8
    res = _validate(cfg)
    assert res["valid"] is False
    assert "local_structure_weights_not_normalized" in _codes(res)


def test_global_clamp_invalid() -> None:
    cfg = _base_cfg()
    cfg["global_metrics"]["clamp"] = [1, 1]
    res = _validate(cfg)
    assert res["valid"] is False
    assert "global_clamp_invalid" in _codes(res)


def test_local_clamp_invalid() -> None:
    cfg = _base_cfg()
    cfg["local_metrics"]["clamp"] = [0, 0]
    res = _validate(cfg)
    assert res["valid"] is False
    assert "local_clamp_invalid" in _codes(res)


def test_synthetic_frames_range_invalid() -> None:
    cfg = _base_cfg()
    cfg["synthetic"]["frames_min"] = 20
    cfg["synthetic"]["frames_max"] = 10
    res = _validate(cfg)
    assert res["valid"] is False
    assert "synthetic_frames_range_invalid" in _codes(res)


def test_data_frames_target_invalid() -> None:
    cfg = _base_cfg()
    cfg["data"]["frames_min"] = 50
    cfg["data"]["frames_target"] = 10
    res = _validate(cfg)
    assert res["valid"] is False
    assert "data_frames_target_invalid" in _codes(res)


def test_cluster_count_range_invalid() -> None:
    cfg = _base_cfg()
    cfg["synthetic"]["clustering"]["cluster_count_range"] = [30, 15]
    res = _validate(cfg)
    assert res["valid"] is False
    assert "cluster_count_range_invalid" in _codes(res)


def test_schema_missing_required_section_is_invalid() -> None:
    cfg = _base_cfg()
    del cfg["runtime_limits"]
    res = _validate(cfg)
    assert res["valid"] is False
    assert "schema_validation_error" in _codes(res)
