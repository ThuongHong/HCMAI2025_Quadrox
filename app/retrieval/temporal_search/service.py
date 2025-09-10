from __future__ import annotations

from dataclasses import asdict
from math import exp
from typing import List, Dict, Tuple, Any, Optional, Callable
import os
from pathlib import Path
import logging

from .io_map import KeyframeMap
from .grouping import KF, cluster_keyframes_by_time
from .abts import Pivot, auto_temporal_window


# Singleton-style map loader for reuse
_KF_MAP: Optional[KeyframeMap] = None
_Scorer: Optional[Callable[[str, List[Tuple[int, float, float]]], List[Tuple[int, float, float]]]] = None


def register_temporal_scorer(fn: Callable[[str, List[Tuple[int, float, float]]], List[Tuple[int, float, float]]]):
    """Optional: register a scorer to re-score neighborhood tuples with query similarity.

    fn(video_id, [(frame_idx, pts_time, score_like)]) -> updated list with rescored 'score_like'.
    """
    global _Scorer
    _Scorer = fn


def _maybe_rescore(video_id: str, neigh: List[Tuple[int, float, float]]) -> List[Tuple[int, float, float]]:
    global _Scorer
    return _Scorer(video_id, neigh) if _Scorer else neigh


def _resolve_map_root() -> Path:
    """Resolve resources/map-keyframes path robustly across run contexts."""
    # 1) Env var override
    env_p = os.environ.get("MAP_KEYFRAMES_ROOT") or os.environ.get("MAP_KEYFRAMES_DIR")
    if env_p:
        p = Path(env_p)
        if p.exists():
            return p
    # 2) Derive repo root from this file location: app/retrieval/temporal_search/service.py -> repo
    repo_root = Path(__file__).resolve().parents[3]
    candidates = [
        repo_root / "resources" / "map-keyframes",
        Path.cwd() / "resources" / "map-keyframes",
        Path.cwd().parent / "resources" / "map-keyframes",
    ]
    seen: set[str] = set()
    uniq = []
    for c in candidates:
        s = str(c.resolve())
        if s not in seen:
            seen.add(s)
            uniq.append(c)
    for c in uniq:
        if c.exists():
            return c
    # Fallback to repo-root default
    return repo_root / "resources" / "map-keyframes"


def _get_map() -> KeyframeMap:
    global _KF_MAP
    if _KF_MAP is None:
        _KF_MAP = KeyframeMap(root=_resolve_map_root())
    return _KF_MAP


def video_id_from_nums(group_num: int, video_num: int, prefix: str = "L") -> str:
    return f"{prefix}{int(group_num):02d}_V{int(video_num):03d}"


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
            normalize=lambda arr: _maybe_rescore(video_id, arr),
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

    # Logging telemetry
    try:
        logger = logging.getLogger(__name__)
        logger.info(
            f"TemporalEnrich mode={mode} pivot=({video_id}, fi={frame_idx}, t={pts_time:.3f}) window=({t_left:.3f},{t_right:.3f}) clusters={sum(len(v) for v in clusters.values())}"
        )
        for vid, cs in clusters.items():
            for c in cs:
                rep = c.representative
                logger.info(
                    f"  {vid}: [{c.start_time:.2f},{c.end_time:.2f}] rep(n={rep.n}, fi={rep.frame_idx}, t={rep.pts_time:.2f}, s={rep.score})"
                )
    except Exception:
        pass

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
