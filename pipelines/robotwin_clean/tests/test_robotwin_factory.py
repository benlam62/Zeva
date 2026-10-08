from __future__ import annotations

import json
from pathlib import Path

import cv2
import h5py
import numpy as np
import pytest
from zeva_robotwin_clean.contract import CAMERA_KEYS
from zeva_robotwin_clean.dataset import validate_prepared_sample
from zeva_robotwin_clean.episodes import RoboTwinCleanWindowDataset
from zeva_robotwin_clean.robotwin_factory import DEFAULT_RAW_DATA_ROOT
from zeva_robotwin_clean.robotwin_factory import load_robotwin_raw_episode
from zeva_robotwin_clean.robotwin_factory import make_datasets
from zeva_robotwin_clean.robotwin_factory import split_task_descriptors


def _create_mock_episode(task_dir: Path, episode_idx: int, length: int = 4) -> None:
    data_dir = task_dir / "aloha_agilex" / "data"
    inst_dir = task_dir / "aloha_agilex" / "instruction"
    data_dir.mkdir(parents=True, exist_ok=True)
    inst_dir.mkdir(parents=True, exist_ok=True)

    h5_path = data_dir / f"episode_{episode_idx:07d}.hdf5"
    inst_path = inst_dir / f"episode_{episode_idx:07d}.json"

    inst_path.write_text(
        json.dumps({"seen": [f"perform task {task_dir.name}"], "unseen": []}),
        encoding="utf-8",
    )

    # Encode a dummy 1x1 image as JPEG
    dummy_img = np.zeros((8, 8, 3), dtype=np.uint8)
    _, encoded = cv2.imencode(".jpg", dummy_img)
    encoded_bytes = encoded.tobytes()

    with h5py.File(h5_path, "w") as f:
        f.create_dataset("state/left_arm_joint_states", data=np.zeros((length, 6), dtype=np.float32))
        f.create_dataset("state/left_ee_joint_states", data=np.ones((length, 1), dtype=np.float32))
        f.create_dataset("state/right_arm_joint_states", data=np.zeros((length, 6), dtype=np.float32))
        f.create_dataset("state/right_ee_joint_states", data=np.zeros((length, 1), dtype=np.float32))

        # EE poses: xyz + xyzw quaternion
        left_ee = np.zeros((length, 7), dtype=np.float32)
        left_ee[:, 6] = 1.0  # qw = 1
        right_ee = np.zeros((length, 7), dtype=np.float32)
        right_ee[:, 6] = 1.0  # qw = 1
        f.create_dataset("state/left_ee_poses", data=left_ee)
        f.create_dataset("state/right_ee_poses", data=right_ee)

        # Image byte streams
        byte_array = np.array([encoded_bytes] * length, dtype=f"|S{len(encoded_bytes)}")
        for key in ("vision/cam_head/colors", "vision/cam_left_wrist/colors", "vision/cam_right_wrist/colors"):
            f.create_dataset(key, data=byte_array)


def test_split_task_descriptors(tmp_path: Path) -> None:
    task_dir1 = tmp_path / "task1"
    task_dir2 = tmp_path / "task2"
    for i in range(10):
        _create_mock_episode(task_dir1, i, length=3)
        _create_mock_episode(task_dir2, i, length=3)

    train_desc, val_desc = split_task_descriptors(tmp_path, eval_split=0.2)
    # 10 episodes per task, 2 tasks = 20 total. With 0.2 split, 2 per task are val, 8 train.
    assert len(train_desc) == 16
    assert len(val_desc) == 4

    assert all(d.split == "Clean" for d in train_desc)
    assert all(d.split == "Clean" for d in val_desc)


def test_make_datasets_mock(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    task_dir = tmp_path / "adjust_bottle"
    for i in range(5):
        _create_mock_episode(task_dir, i, length=4)

    monkeypatch.setenv("ROBOTWIN_RAW_DATA_ROOT", str(tmp_path))

    train_ds, val_ds = make_datasets({"dataset": {"eval_split": 0.2}})
    assert isinstance(train_ds, RoboTwinCleanWindowDataset)
    assert isinstance(val_ds, RoboTwinCleanWindowDataset)

    assert train_ds.num_episodes == 4
    assert val_ds.num_episodes == 1

    # Verify LeRobot metadata contract
    assert train_ds.meta.camera_keys == list(CAMERA_KEYS)
    assert not train_ds.meta.has_language_columns
    assert train_ds.meta.episodes["dataset_from_index"] == [0, 4, 8, 12]
    assert train_ds.meta.episodes["dataset_to_index"] == [4, 8, 12, 16]
    for key in CAMERA_KEYS:
        assert train_ds.meta.features[key]["names"] == ["height", "width", "channels"]

    sample = train_ds[0]
    validate_prepared_sample(sample)
    assert sample["observation.state"].shape == (14,)
    assert sample["action"].shape == (50, 16)
    assert sample[CAMERA_KEYS[0]].shape == (3, 480, 640)

    val_sample = val_ds[0]
    validate_prepared_sample(val_sample)


def test_simple_dataset_metadata_filters_string_fields() -> None:
    from zeva_robotwin_clean.robotwin_factory import SimpleDatasetMetadata

    raw_stats = {
        "schema": "zeva-ego-robotwin-normalization-v1",
        "subset": "train",
        "action": {"mean": [0.0] * 16, "std": [1.0] * 16},
        "observation.state": {"mean": [0.0] * 14, "std": [1.0] * 14},
    }
    meta = SimpleDatasetMetadata(stats=raw_stats)
    assert "schema" not in meta.stats
    assert "subset" not in meta.stats
    assert "action" in meta.stats
    assert "observation.state" in meta.stats


@pytest.mark.skipif(
    not DEFAULT_RAW_DATA_ROOT.is_dir(),
    reason="Raw RoboTwin clean dataset not present on host",
)
def test_real_raw_dataset_smoke() -> None:
    train_desc, val_desc = split_task_descriptors(DEFAULT_RAW_DATA_ROOT, eval_split=0.1)
    assert len(train_desc) > 0
    assert len(val_desc) > 0

    # Test loading one sample from train and one from val
    train_ds = RoboTwinCleanWindowDataset([train_desc[0]], load_robotwin_raw_episode)
    val_ds = RoboTwinCleanWindowDataset([val_desc[0]], load_robotwin_raw_episode)

    sample_train = train_ds[0]
    validate_prepared_sample(sample_train)
    assert sample_train["observation.state"].shape == (14,)
    assert sample_train["action"].shape == (50, 16)

    sample_val = val_ds[0]
    validate_prepared_sample(sample_val)
    assert sample_val["observation.state"].shape == (14,)
    assert sample_val["action"].shape == (50, 16)
