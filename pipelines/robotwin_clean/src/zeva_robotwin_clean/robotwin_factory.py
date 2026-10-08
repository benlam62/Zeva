"""RoboTwin demo_clean dataset factory for LeRobot training and evaluation."""

from __future__ import annotations

from collections.abc import Sequence
import json
import os
from pathlib import Path
from typing import Any

import cv2
import h5py
import numpy as np

from zeva_robotwin_clean.episodes import EpisodeArrays
from zeva_robotwin_clean.episodes import EpisodeDescriptor
from zeva_robotwin_clean.episodes import RoboTwinCleanWindowDataset

DEFAULT_RAW_DATA_ROOT = Path("/home/benlam/RoboTwin-202610/data/zeva")
TARGET_RESOLUTION = (640, 480)  # (width, height) to match image shape [480, 640, 3]


class SimpleDatasetMetadata:
    """Attaches training-split statistics and features schema to the dataset."""

    def __init__(
        self,
        stats: dict[str, Any] | None = None,
        episodes: dict[str, list[int]] | None = None,
    ) -> None:
        self.stats = {k: v for k, v in stats.items() if isinstance(v, dict)} if stats else {}
        self.episodes = episodes or {}
        self.features = {
            "observation.state": {"dtype": "float32", "shape": [14]},
            "action": {"dtype": "float32", "shape": [16]},
            "observation.images.cam_high": {
                "dtype": "image",
                "shape": [480, 640, 3],
                "names": ["height", "width", "channels"],
            },
            "observation.images.cam_left_wrist": {
                "dtype": "image",
                "shape": [480, 640, 3],
                "names": ["height", "width", "channels"],
            },
            "observation.images.cam_right_wrist": {
                "dtype": "image",
                "shape": [480, 640, 3],
                "names": ["height", "width", "channels"],
            },
        }

    @property
    def camera_keys(self) -> list[str]:
        return [key for key, ft in self.features.items() if ft["dtype"] in ["video", "image"]]

    @property
    def has_language_columns(self) -> bool:
        return False


def load_robotwin_raw_episode(descriptor: EpisodeDescriptor) -> EpisodeArrays:
    """Decode Joint14, absolute EEF16, and three camera views for one episode."""
    h5_path = Path(descriptor.reference)
    with h5py.File(h5_path, "r") as f:
        # Joint14: [T, 14]
        l_arm = f["state/left_arm_joint_states"][:]
        l_grip = f["state/left_ee_joint_states"][:]
        r_arm = f["state/right_arm_joint_states"][:]
        r_grip = f["state/right_ee_joint_states"][:]
        joint14 = np.concatenate([l_arm, l_grip, r_arm, r_grip], axis=-1).astype(np.float32)

        # Absolute EEF16: [T, 16] (left xyz/xyzw/gripper, right xyz/xyzw/gripper)
        l_ee = f["state/left_ee_poses"][:]
        r_ee = f["state/right_ee_poses"][:]
        absolute_eef16 = np.concatenate(
            [
                l_ee[:, :3],
                l_ee[:, 3:7],
                l_grip,
                r_ee[:, :3],
                r_ee[:, 3:7],
                r_grip,
            ],
            axis=-1,
        ).astype(np.float32)

        # Decode JPEG/PNG camera streams
        def decode_stream(key: str) -> np.ndarray:
            raw_bytes = f[key][:]
            frames = []
            for b in raw_bytes:
                img_bgr = cv2.imdecode(np.frombuffer(b, np.uint8), cv2.IMREAD_COLOR)
                img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
                if TARGET_RESOLUTION and (img_rgb.shape[1], img_rgb.shape[0]) != TARGET_RESOLUTION:
                    img_rgb = cv2.resize(img_rgb, TARGET_RESOLUTION)
                frames.append(img_rgb)
            return np.stack(frames, axis=0)

        images = {
            "observation.images.cam_high": decode_stream("vision/cam_head/colors"),
            "observation.images.cam_left_wrist": decode_stream("vision/cam_left_wrist/colors"),
            "observation.images.cam_right_wrist": decode_stream("vision/cam_right_wrist/colors"),
        }

    return EpisodeArrays(joint14=joint14, absolute_eef16=absolute_eef16, images=images)


def split_task_descriptors(
    clean_root: Path | str = DEFAULT_RAW_DATA_ROOT,
    eval_split: float = 0.1,
) -> tuple[list[EpisodeDescriptor], list[EpisodeDescriptor]]:
    """Index episodes per task and split into train and validation sets."""
    clean_root = Path(clean_root)
    train_descriptors: list[EpisodeDescriptor] = []
    val_descriptors: list[EpisodeDescriptor] = []

    for task_dir in sorted(clean_root.iterdir()):
        if not task_dir.is_dir():
            continue
        data_dir = task_dir / "aloha_agilex" / "data"
        inst_dir = task_dir / "aloha_agilex" / "instruction"
        if not data_dir.is_dir():
            continue

        h5_files = sorted(data_dir.glob("*.hdf5"))
        if not h5_files:
            continue

        n_val = int(len(h5_files) * eval_split) if eval_split > 0 else 0
        train_files = h5_files[:-n_val] if n_val > 0 else h5_files
        val_files = h5_files[-n_val:] if n_val > 0 else []

        def build_descriptors(files: list[Path], current_inst_dir: Path, current_task_name: str) -> list[EpisodeDescriptor]:
            descriptors = []
            for h5_file in files:
                inst_file = current_inst_dir / f"{h5_file.stem}.json"
                instruction = "perform the task"
                if inst_file.is_file():
                    with open(inst_file, encoding="utf-8") as jf:
                        instructions = json.load(jf)
                        seen = instructions.get("seen", [])
                        if seen:
                            instruction = seen[0]

                with h5py.File(h5_file, "r") as hf:
                    length = hf["state/left_arm_joint_states"].shape[0]

                descriptors.append(
                    EpisodeDescriptor(
                        reference=str(h5_file.resolve()),
                        split="Clean",
                        task=current_task_name,
                        instruction=instruction,
                        length=length,
                    )
                )
            return descriptors

        train_descriptors.extend(build_descriptors(train_files, inst_dir, task_dir.name))
        val_descriptors.extend(build_descriptors(val_files, inst_dir, task_dir.name))

    return train_descriptors, val_descriptors


def make_datasets(
    train_config: Any,
) -> tuple[RoboTwinCleanWindowDataset, RoboTwinCleanWindowDataset | None]:
    """LeRobot dataset factory returning (train_dataset, validation_dataset)."""
    clean_root = Path(os.environ.get("ROBOTWIN_RAW_DATA_ROOT", DEFAULT_RAW_DATA_ROOT))

    eval_split = 0.1
    if hasattr(train_config, "dataset") and hasattr(train_config.dataset, "eval_split"):
        eval_split = float(train_config.dataset.eval_split)
    elif isinstance(train_config, dict):
        eval_split = float(train_config.get("dataset", {}).get("eval_split", 0.1))

    train_desc, val_desc = split_task_descriptors(clean_root, eval_split=eval_split)

    stats_path_env = os.environ.get("ROBOTWIN_STATS_PATH", "outputs/robotwin-clean-stats.json")
    stats_file = Path(stats_path_env)
    stats = json.loads(stats_file.read_text(encoding="utf-8")) if stats_file.is_file() else None

    def _make_meta(descriptors: Sequence[EpisodeDescriptor]) -> SimpleDatasetMetadata:
        ends: list[int] = []
        total = 0
        for d in descriptors:
            total += d.length
            ends.append(total)
        episodes_meta = {
            "dataset_from_index": [0] + ends[:-1] if ends else [],
            "dataset_to_index": ends,
        }
        return SimpleDatasetMetadata(stats=stats, episodes=episodes_meta)

    prepared_train_dataset = RoboTwinCleanWindowDataset(
        descriptors=train_desc,
        episode_loader=load_robotwin_raw_episode,
        meta=_make_meta(train_desc),
    )

    prepared_validation_dataset = (
        RoboTwinCleanWindowDataset(
            descriptors=val_desc,
            episode_loader=load_robotwin_raw_episode,
            meta=_make_meta(val_desc),
        )
        if val_desc
        else None
    )

    return prepared_train_dataset, prepared_validation_dataset
