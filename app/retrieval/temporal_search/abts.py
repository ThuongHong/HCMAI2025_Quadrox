from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, List, Tuple
import numpy as np


@dataclass
class Pivot:
    video_id: str
    frame_idx: int
    pts_time: float
    score: float | None = None


@dataclass
class Window:
    start_time: float
    end_time: float


def auto_temporal_window(
    pivot: Pivot,
    neighborhood_fetch: Callable[[str, float, float], List[Tuple[int, float, float]]],
    normalize: Callable[[List[Tuple[int, float, float]]], List[Tuple[int, float, float]]] | None = None,
    init_delta: float = 5.0,
    max_delta: float = 20.0,
    step: float = 2.5,
    stability_k: int = 2,
    min_conf_drop: float = 0.15,
) -> Window:
    """
    Expand ±delta around pivot until similarity/stability drops enough.
    Simplified ABTS: only uses neighborhood scores; if embeddings are available,
    neighborhood_fetch should compute/attach fresh similarity; else reuse scores.
    Returns a time window [start_time, end_time].
    """
    delta = init_delta
    best = Window(pivot.pts_time - delta, pivot.pts_time + delta)

    prev_conf = None
    while delta < max_delta:
        left = pivot.pts_time - delta
        right = pivot.pts_time + delta
        neigh = neighborhood_fetch(pivot.video_id, left, right)
        if not neigh:
            break
        if normalize:
            neigh = normalize(neigh)

        # edge confidence ~ average of top-K scores near edges
        left_edge = sorted(
            [s for (_fi, t, s) in neigh if left <= t <= left + step],
            key=lambda x: x,
            reverse=True,
        )[:stability_k]
        right_edge = sorted(
            [s for (_fi, t, s) in neigh if right - step <= t <= right],
            key=lambda x: x,
            reverse=True,
        )[:stability_k]

        def avg(xs):
            return float(np.mean(xs)) if xs else 0.0

        edge_conf = 0.5 * (avg(left_edge) + avg(right_edge))

        if prev_conf is None or (prev_conf - edge_conf) < min_conf_drop:
            best = Window(left, right)
            prev_conf = edge_conf
            delta += step
        else:
            break
    return best

