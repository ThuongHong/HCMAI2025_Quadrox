from __future__ import annotations

from dataclasses import dataclass
from typing import List, Dict, Any
import math


@dataclass
class KF:
    video_id: str
    n: int | None
    frame_idx: int
    pts_time: float
    score: float | None = None
    raw: Dict[str, Any] | None = None


@dataclass
class TemporalCluster:
    video_id: str
    start_time: float
    end_time: float
    kfs: List[KF]
    representative: KF


def _finalize_cluster(vid: str, arr: List[KF]) -> TemporalCluster:
    start_t = arr[0].pts_time
    end_t = arr[-1].pts_time
    # choose representative: max score if available else prefer middle
    center = (start_t + end_t) / 2.0
    rep = max(
        arr,
        key=lambda k: (
            k.score if k.score is not None else -math.inf,
            -abs(k.pts_time - center),
        ),
    )
    return TemporalCluster(
        video_id=vid, start_time=start_t, end_time=end_t, kfs=arr, representative=rep
    )


def cluster_keyframes_by_time(
    items: List[KF], gap_seconds: float = 10.0
) -> Dict[str, List[TemporalCluster]]:
    """
    Group keyframes for each video into time clusters.
    A new cluster starts when gap to previous keyframe's pts_time > gap_seconds.
    """
    by_vid: Dict[str, List[KF]] = {}
    for x in items:
        by_vid.setdefault(x.video_id, []).append(x)

    out: Dict[str, List[TemporalCluster]] = {}
    for vid, arr in by_vid.items():
        arr.sort(key=lambda k: (k.pts_time, k.frame_idx))
        clusters: List[TemporalCluster] = []
        cur: List[KF] = []
        for k in arr:
            if not cur:
                cur = [k]
                continue
            if k.pts_time - cur[-1].pts_time > gap_seconds:
                clusters.append(_finalize_cluster(vid, cur))
                cur = [k]
            else:
                cur.append(k)
        if cur:
            clusters.append(_finalize_cluster(vid, cur))
        out[vid] = clusters
    return out

