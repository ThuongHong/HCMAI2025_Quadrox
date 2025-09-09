from __future__ import annotations

from dataclasses import asdict
from math import exp
from typing import List, Dict, Tuple, Any, Optional

from .io_map import KeyframeMap
from .grouping import KF, cluster_keyframes_by_time
from .abts import Pivot, auto_temporal_window


# Singleton-style map loader for reuse
_KF_MAP: Optional[KeyframeMap] = None


def _get_map() -> KeyframeMap:
    global _KF_MAP
    if _KF_MAP is None:
        _KF_MAP = KeyframeMap(root="resources/map-keyframes")
    return _KF_MAP


def video_id_from_nums(group_num: int, video_num: int) -> str:
    return f"L{int(group_num):02d}_V{int(video_num):03d}"


def _fetch_neighborhood_scores(
    video_id: str,
    t_left: float,
    t_right: float,
    pivot_time: float,
    tau: float = 3.0,
) -> List[Tuple[int, float, float]]:
    """
    Return [(frame_idx, pts_time, score_like)] for keyframes within [t_left, t_right].
    Score uses temporal decay from pivot_time as a fallback.
    """
    kfmap = _get_map()
    df = kfmap.load(video_id)
    df2 = df[(df["pts_time"] >= t_left) & (df["pts_time"] <= t_right)].sort_values("pts_time")
    out: List[Tuple[int, float, float]] = []
    for _, r in df2.iterrows():
        t = float(r.pts_time)
        # temporal decay around pivot
        s = exp(-abs(t - pivot_time) / max(tau, 1e-6))
        out.append((int(r.frame_idx), t, float(s)))
    return out


def _fetch_neighborhood_full(
    video_id: str,
    t_left: float,
    t_right: float,
    pivot_time: float,
    tau: float = 3.0,
) -> List[Tuple[int, float, float, int]]:
    """
    Return [(frame_idx, pts_time, score_like, n)] for keyframes within [t_left, t_right].
    """
    kfmap = _get_map()
    df = kfmap.load(video_id)
    df2 = df[(df["pts_time"] >= t_left) & (df["pts_time"] <= t_right)].sort_values("pts_time")
    out: List[Tuple[int, float, float, int]] = []
    for _, r in df2.iterrows():
        t = float(r.pts_time)
        s = exp(-abs(t - pivot_time) / max(tau, 1e-6))
        out.append((int(r.frame_idx), t, float(s), int(r.n)))
    return out


def _ensure_pivot_time(video_id: str, n: Optional[int], frame_idx: Optional[int], pts_time: Optional[float]) -> Tuple[int, float]:
    """Return (frame_idx, pts_time) for pivot, using CSV mapping where needed."""
    kfmap = _get_map()
    if pts_time is not None and frame_idx is not None:
        return int(frame_idx), float(pts_time)
    if n is not None:
        m = kfmap.n_to_frame(video_id, int(n))
        return int(m["frame_idx"]), float(m["pts_time"])
    # If only frame_idx provided, approximate pts_time by nearest in CSV
    if frame_idx is not None:
        df = kfmap.load(video_id)
        row = df.loc[df["frame_idx"] == int(frame_idx)].head(1)
        if row.empty:
            # nearest
            row = df.iloc[(df["frame_idx"] - int(frame_idx)).abs().argsort()].head(1)
        r = row.iloc[0]
        return int(r["frame_idx"]), float(r["pts_time"])
    raise ValueError("Insufficient pivot information: provide at least n or (frame_idx & pts_time)")


def temporal_enrich(
    mode: str,
    video_id: str,
    pivot_n: Optional[int] = None,
    pivot_frame_idx: Optional[int] = None,
    pivot_pts_time: Optional[float] = None,
    pivot_score: Optional[float] = None,
    delta: float = 5.0,
    gap_seconds: float = 10.0,
) -> Dict[str, Any]:
    """
    Build a temporal view model from a pivot.

    Returns dict with fields:
    - mode, pivot, clusters[{video_id,start_time,end_time,representative{frame_idx,pts_time,score,n},keyframes[{frame_idx,pts_time,score,n}]}]
    """
    assert mode in {"auto", "interactive"}
    frame_idx, pts_time = _ensure_pivot_time(video_id, pivot_n, pivot_frame_idx, pivot_pts_time)

    # Determine window
    if mode == "auto":
        win = auto_temporal_window(
            pivot=Pivot(video_id=video_id, frame_idx=frame_idx, pts_time=pts_time, score=pivot_score),
            neighborhood_fetch=lambda vid, l, r: _fetch_neighborhood_scores(vid, l, r, pts_time),
            init_delta=5.0,
            max_delta=20.0,
            step=2.5,
        )
        t_left, t_right = win.start_time, win.end_time
    else:
        d = max(1.0, float(delta))
        t_left, t_right = pts_time - d, pts_time + d

    # Collect items inside window and cluster
    neigh = _fetch_neighborhood_full(video_id, t_left, t_right, pts_time)
    items: List[KF] = [
        KF(video_id=video_id, n=n, frame_idx=fi, pts_time=t, score=s) for (fi, t, s, n) in neigh
    ]
    clusters = cluster_keyframes_by_time(items, gap_seconds=gap_seconds)

    # Build view model
    out_clusters: List[Dict[str, Any]] = []
    for vid, cs in clusters.items():
        for c in cs:
            out_clusters.append(
                {
                    "video_id": vid,
                    "start_time": c.start_time,
                    "end_time": c.end_time,
                    "representative": {
                        "frame_idx": c.representative.frame_idx,
                        "pts_time": c.representative.pts_time,
                        "score": c.representative.score,
                        "n": c.representative.n,
                    },
                    "keyframes": [
                        {
                            "frame_idx": k.frame_idx,
                            "pts_time": k.pts_time,
                            "score": k.score,
                            "n": k.n,
                        }
                        for k in c.kfs
                    ],
                }
            )

    return {
        "mode": mode,
        "pivot": {
            "video_id": video_id,
            "frame_idx": frame_idx,
            "pts_time": pts_time,
            "score": pivot_score,
        },
        "window": {"start_time": t_left, "end_time": t_right},
        "clusters": out_clusters,
    }

