"""Train-split normalization statistics for RoboTwin clean post-training."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any

import numpy as np
import torch

FEATURES = ("observation.state", "action")
DEFAULT_QUANTILES = (0.01, 0.10, 0.50, 0.90, 0.99)


class _RunningQuantileStatsFallback:
    """Streaming quantile estimator matching LeRobot's histogram-based algorithm."""

    def __init__(
        self,
        quantile_list: tuple[float, ...] = DEFAULT_QUANTILES,
        num_quantile_bins: int = 5000,
    ) -> None:
        self._count = 0
        self._min: np.ndarray | None = None
        self._max: np.ndarray | None = None
        self._histograms: list[np.ndarray] | None = None
        self._bin_edges: list[np.ndarray] | None = None
        self._num_quantile_bins = num_quantile_bins
        self._quantile_list = quantile_list
        self._quantile_keys = [f"q{int(q * 100):02d}" for q in self._quantile_list]

    def update(self, batch: np.ndarray) -> None:
        flat = batch.reshape(-1, batch.shape[-1]).astype(np.float32, copy=False)
        num_elements, vector_length = flat.shape
        if self._count == 0:
            self._min = np.min(flat, axis=0)
            self._max = np.max(flat, axis=0)
            self._histograms = [np.zeros(self._num_quantile_bins) for _ in range(vector_length)]
            self._bin_edges = [
                np.linspace(self._min[i] - 1e-10, self._max[i] + 1e-10, self._num_quantile_bins + 1)
                for i in range(vector_length)
            ]
        else:
            new_max = np.max(flat, axis=0)
            new_min = np.min(flat, axis=0)
            max_changed = np.any(new_max > self._max)
            min_changed = np.any(new_min < self._min)
            self._max = np.maximum(self._max, new_max)
            self._min = np.minimum(self._min, new_min)
            if max_changed or min_changed:
                self._adjust_histograms()

        self._count += num_elements
        for i in range(vector_length):
            hist, _ = np.histogram(flat[:, i], bins=self._bin_edges[i])
            self._histograms[i] += hist

    def _adjust_histograms(self) -> None:
        for i in range(len(self._histograms)):
            old_edges = self._bin_edges[i]
            old_hist = self._histograms[i]
            padding = (self._max[i] - self._min[i]) * 1e-10
            new_edges = np.linspace(
                self._min[i] - padding,
                self._max[i] + padding,
                self._num_quantile_bins + 1,
            )
            old_centers = (old_edges[:-1] + old_edges[1:]) / 2.0
            new_hist = np.zeros(self._num_quantile_bins)
            for old_center, count in zip(old_centers, old_hist, strict=False):
                if count > 0:
                    bin_idx = int(np.searchsorted(new_edges, old_center) - 1)
                    bin_idx = max(0, min(bin_idx, self._num_quantile_bins - 1))
                    new_hist[bin_idx] += count
            self._histograms[i] = new_hist
            self._bin_edges[i] = new_edges

    def get_statistics(self) -> dict[str, np.ndarray]:
        results: list[np.ndarray] = []
        for q in self._quantile_list:
            target_count = q * self._count
            q_values: list[float] = []
            for hist, edges in zip(self._histograms, self._bin_edges, strict=True):
                cumsum = np.cumsum(hist)
                idx = int(np.searchsorted(cumsum, target_count))
                if idx == 0:
                    val = float(edges[0])
                elif idx >= len(cumsum):
                    val = float(edges[-1])
                else:
                    count_before = cumsum[idx - 1]
                    count_in_bin = cumsum[idx] - count_before
                    if count_in_bin == 0:
                        val = float(edges[idx])
                    else:
                        fraction = (target_count - count_before) / count_in_bin
                        val = float(edges[idx] + fraction * (edges[idx + 1] - edges[idx]))
                q_values.append(val)
            results.append(np.array(q_values))
        return {q_key: results[i] for i, q_key in enumerate(self._quantile_keys)}


def _get_quantile_accumulator_cls() -> type:
    try:
        from lerobot.datasets.compute_stats import RunningQuantileStats
        return RunningQuantileStats
    except (ImportError, AttributeError):
        return _RunningQuantileStatsFallback


def compute_train_statistics(samples: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    """Compute the statistics consumed by LeRobot's QUANTILES normalizer.

    The iterable must contain only training samples. Quantiles are accumulated
    with LeRobot's streaming quantile estimator (falling back to a built-in
    exact implementation if lerobot.datasets is not installed).
    """

    accumulator_cls = _get_quantile_accumulator_cls()
    accumulators = {key: accumulator_cls() for key in FEATURES}
    totals: dict[str, np.ndarray] = {}
    squares: dict[str, np.ndarray] = {}
    minima: dict[str, np.ndarray] = {}
    maxima: dict[str, np.ndarray] = {}
    counts = dict.fromkeys(FEATURES, 0)
    for sample in samples:
        for key in FEATURES:
            tensor = sample[key]
            if not isinstance(tensor, torch.Tensor) or not tensor.is_floating_point():
                raise TypeError(f"{key} must be a floating tensor")
            values = tensor.detach().cpu().numpy().astype(np.float32, copy=False)
            flat = values.reshape(-1, values.shape[-1]).astype(np.float64)
            if not np.isfinite(flat).all():
                raise ValueError(f"{key} contains non-finite values")
            if key not in totals:
                totals[key] = np.zeros(flat.shape[-1], dtype=np.float64)
                squares[key] = np.zeros(flat.shape[-1], dtype=np.float64)
                minima[key] = np.full(flat.shape[-1], np.inf)
                maxima[key] = np.full(flat.shape[-1], -np.inf)
            counts[key] += len(flat)
            totals[key] += flat.sum(axis=0)
            squares[key] += np.square(flat).sum(axis=0)
            minima[key] = np.minimum(minima[key], flat.min(axis=0))
            maxima[key] = np.maximum(maxima[key], flat.max(axis=0))
            accumulators[key].update(flat.astype(np.float32, copy=False))
    result: dict[str, Any] = {"schema": "zeva-ego-robotwin-normalization-v1", "subset": "train"}
    for key in FEATURES:
        if counts[key] == 0:
            raise ValueError("the training sample iterable is empty")
        mean = totals[key] / counts[key]
        variance = np.maximum(squares[key] / counts[key] - np.square(mean), 0.0)
        quantiles = accumulators[key].get_statistics()
        result[key] = {
            "count": counts[key],
            "mean": mean.tolist(),
            "std": np.sqrt(variance).tolist(),
            "min": minima[key].tolist(),
            "max": maxima[key].tolist(),
            **{name: np.asarray(quantiles[name]).tolist() for name in ("q01", "q10", "q50", "q90", "q99")},
        }
    return result


def quantile_normalize(values: torch.Tensor, statistics: Mapping[str, Any]) -> torch.Tensor:
    """Map the 1st--99th percentile interval to [-1,1]."""

    lower = values.new_tensor(statistics["q01"])
    upper = values.new_tensor(statistics["q99"])
    scale = torch.clamp(upper - lower, min=1e-6)
    return 2.0 * (values - lower) / scale - 1.0


def quantile_unnormalize(values: torch.Tensor, statistics: Mapping[str, Any]) -> torch.Tensor:
    lower = values.new_tensor(statistics["q01"])
    upper = values.new_tensor(statistics["q99"])
    return (values + 1.0) * 0.5 * (upper - lower) + lower
