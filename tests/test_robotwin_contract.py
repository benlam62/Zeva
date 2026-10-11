from __future__ import annotations

import pytest

from openpi.zeva.robotwin_contract import ROBOTWIN_ACTION_DIM
from openpi.zeva.robotwin_contract import ROBOTWIN_ACTION_FEATURE_NAMES
from openpi.zeva.robotwin_contract import ROBOTWIN_ACTION_HORIZON
from openpi.zeva.robotwin_contract import ROBOTWIN_ACTION_NAMES
from openpi.zeva.robotwin_contract import ROBOTWIN_CAMERA_KEYS
from openpi.zeva.robotwin_contract import ROBOTWIN_IMAGE_SHAPE
from openpi.zeva.robotwin_contract import ROBOTWIN_STATE_DIM
from openpi.zeva.robotwin_contract import RobotWinHandoff


def _valid_base_config() -> dict:
    return {
        "type": "pi05",
        "chunk_size": ROBOTWIN_ACTION_HORIZON,
        "n_action_steps": ROBOTWIN_ACTION_HORIZON,
        "input_features": {
            "observation.state": {"shape": [ROBOTWIN_STATE_DIM]},
            **{key: {"shape": list(ROBOTWIN_IMAGE_SHAPE)} for key in ROBOTWIN_CAMERA_KEYS},
        },
        "output_features": {
            "action": {"shape": [ROBOTWIN_ACTION_DIM]},
        },
        "normalization_mapping": {
            "ACTION": "MEAN_STD",
            "STATE": "MEAN_STD",
            "VISUAL": "IDENTITY",
        },
        "action_feature_names": list(ROBOTWIN_ACTION_NAMES),
    }


def test_validate_checkpoint_config_accepts_mean_std() -> None:
    config = _valid_base_config()
    RobotWinHandoff._validate_checkpoint_config(config)  # noqa: SLF001


def test_validate_checkpoint_config_accepts_quantiles() -> None:
    config = _valid_base_config()
    config["normalization_mapping"] = {
        "ACTION": "QUANTILES",
        "STATE": "QUANTILES",
        "VISUAL": "IDENTITY",
    }
    config["action_feature_names"] = list(ROBOTWIN_ACTION_FEATURE_NAMES)
    RobotWinHandoff._validate_checkpoint_config(config)  # noqa: SLF001


def test_validate_checkpoint_config_rejects_invalid_norm_mapping() -> None:
    config = _valid_base_config()
    config["normalization_mapping"] = {
        "ACTION": "MIN_MAX",
        "STATE": "MIN_MAX",
        "VISUAL": "IDENTITY",
    }
    with pytest.raises(ValueError, match="processor normalization"):
        RobotWinHandoff._validate_checkpoint_config(config)  # noqa: SLF001


def test_validate_checkpoint_config_rejects_invalid_action_names() -> None:
    config = _valid_base_config()
    config["action_feature_names"] = ["invalid_slot"]
    with pytest.raises(ValueError, match="slot order"):
        RobotWinHandoff._validate_checkpoint_config(config)  # noqa: SLF001
