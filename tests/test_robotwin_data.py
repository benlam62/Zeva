from __future__ import annotations

from pathlib import Path
import sys

import torch

from openpi.zeva.robotwin_contract import ROBOTWIN_ACTION_DIM
from openpi.zeva.robotwin_contract import ROBOTWIN_ACTION_HORIZON
from openpi.zeva.robotwin_contract import ROBOTWIN_CAMERA_KEYS
from openpi.zeva.robotwin_contract import ROBOTWIN_IMAGE_SHAPE
from openpi.zeva.robotwin_contract import ROBOTWIN_STATE_DIM
from openpi.zeva.robotwin_data import RoboTwinLeRobotEEF16Dataset
from openpi.zeva.robotwin_data import TorchCodecRoboTwinDataset


def test_egoscale_alias_registered() -> None:
    assert "egoscale" in sys.modules
    from egoscale.data.robotwin.lerobot import RoboTwinLeRobotEEF16Dataset as AliasDataset

    assert AliasDataset is RoboTwinLeRobotEEF16Dataset


def test_robotwin_dataset_loading() -> None:
    manifest_path = Path("/home/benlam/RoboTwin-202610/data/zeva/adapter.json")
    if not manifest_path.is_file():
        return

    train_ds = RoboTwinLeRobotEEF16Dataset(manifest_path, subset="train")
    val_ds = RoboTwinLeRobotEEF16Dataset(manifest_path, subset="validation")

    assert len(train_ds._records) > 0  # noqa: SLF001
    assert len(val_ds._records) > 0  # noqa: SLF001

    # Check non-overlap
    train_ids = {(*r["key"], r["episode_index"]) for r in train_ds._records}  # noqa: SLF001
    val_ids = {(*r["key"], r["episode_index"]) for r in val_ds._records}  # noqa: SLF001
    assert not (train_ids & val_ids)

    # Check item
    sample = train_ds[0]
    assert sample["action"].shape == (ROBOTWIN_ACTION_HORIZON, ROBOTWIN_ACTION_DIM)
    assert sample["observation.state"].shape == (ROBOTWIN_STATE_DIM,)
    assert isinstance(sample["task"], str)


def test_torchcodec_read_images() -> None:
    manifest_path = Path("/home/benlam/RoboTwin-202610/data/zeva/adapter.json")
    if not manifest_path.is_file():
        return

    source = TorchCodecRoboTwinDataset(manifest_path, subset="train")
    record = source.dataset._records[0]  # noqa: SLF001
    images = source.read_images(record, [0, 15])

    for key in ROBOTWIN_CAMERA_KEYS:
        assert key in images
        assert images[key].shape == (2, 3, ROBOTWIN_IMAGE_SHAPE[0], ROBOTWIN_IMAGE_SHAPE[1])
        assert images[key].dtype == torch.uint8
