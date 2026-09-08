"""Lightweight YAML schema checks for ml/config — no torch, no images."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

_REPO = Path(__file__).resolve().parents[2]
_CONFIG_DIR = _REPO / "ml" / "config"

EXPECTED_MODELS = {
    "efficientnet_b0.yaml": "efficientnet_b0",
    "densenet121.yaml": "densenet121",
    "resnet50.yaml": "resnet50",
    "mobilenet_v3.yaml": "mobilenet_v3_small",
}

REQUIRED_MODEL_KEYS = {"name", "pretrained", "num_classes", "dropout"}
REQUIRED_TRAIN_KEYS = {
    "epochs",
    "freeze_epochs",
    "batch_size",
    "num_workers",
    "lr_head",
    "lr_backbone",
    "weight_decay",
    "early_stopping_patience",
    "early_stopping_metric",
    "seed",
    "image_size",
    "mixed_precision",
}
REQUIRED_PATH_KEYS = {
    "data_raw",
    "splits_dir",
    "checkpoint_dir",
    "metrics_dir",
    "plots_dir",
}


@pytest.fixture(scope="module")
def configs() -> dict[str, dict]:
    loaded: dict[str, dict] = {}
    for filename in EXPECTED_MODELS:
        path = _CONFIG_DIR / filename
        assert path.is_file(), f"missing config {path}"
        with path.open(encoding="utf-8") as handle:
            loaded[filename] = yaml.safe_load(handle)
    return loaded


def test_four_backbone_configs_exist(configs: dict[str, dict]) -> None:
    assert set(configs) == set(EXPECTED_MODELS)


def test_model_names_match_contract(configs: dict[str, dict]) -> None:
    for filename, expected_name in EXPECTED_MODELS.items():
        model = configs[filename]["model"]
        assert set(REQUIRED_MODEL_KEYS) <= set(model)
        assert model["name"] == expected_name
        assert model["num_classes"] == 2
        assert model["pretrained"] is True
        assert model["dropout"] == 0.2


def test_train_schedule_matches_contract(configs: dict[str, dict]) -> None:
    for filename, cfg in configs.items():
        train = cfg["train"]
        missing = REQUIRED_TRAIN_KEYS - set(train)
        assert not missing, f"{filename} missing train keys: {missing}"
        assert train["epochs"] == 30
        assert train["freeze_epochs"] == 5
        assert train["batch_size"] == 16
        assert train["seed"] == 42
        assert train["image_size"] == 224
        assert train["early_stopping_metric"] == "val_recall_monkeypox"
        assert train["mixed_precision"] is False


def test_artifact_paths_match_contract(configs: dict[str, dict]) -> None:
    for filename, cfg in configs.items():
        paths = cfg["paths"]
        missing = REQUIRED_PATH_KEYS - set(paths)
        assert not missing, f"{filename} missing path keys: {missing}"
        assert paths["checkpoint_dir"] == "ml/artifacts/checkpoints"
        assert paths["metrics_dir"] == "ml/artifacts/metrics"
        assert paths["splits_dir"] == "data/splits"
