from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional
import pandas as pd


class KeyframeMap:
    """
    Lazy loader for map-keyframes CSVs: <video_id>.csv -> DataFrame with columns:
    n, pts_time, fps, frame_idx

    - video_id format assumed like 'L01_V001'
    - robust to L/K prefix swaps (tries alternate)
    - caches per-path DataFrames
    """

    def __init__(self, root: str | Path):
        self.root = Path(root)
        self._cache: Dict[str, pd.DataFrame] = {}

    def _resolve_path(self, video_id: str) -> Optional[Path]:
        p = self.root / f"{video_id}.csv"
        if p.exists():
            return p
        # try alternate prefix mapping 'L' <-> 'K'
        name = p.name
        if name.startswith("L"):
            alt = name.replace("L", "K", 1)
        elif name.startswith("K"):
            alt = name.replace("K", "L", 1)
        else:
            alt = name
        p2 = self.root / alt
        if p2.exists():
            return p2
        return None

    def load(self, video_id: str) -> pd.DataFrame:
        path = self._resolve_path(video_id)
        if path is None:
            raise FileNotFoundError(f"map-keyframes CSV not found for {video_id} under {self.root}")
        key = str(path)
        if key in self._cache:
            return self._cache[key]
        df = pd.read_csv(path)
        req = {"n", "pts_time", "fps", "frame_idx"}
        if not req.issubset(set(df.columns)):
            raise AssertionError(
                f"map-keyframes schema mismatch for {video_id}: expected columns {sorted(req)}; got {sorted(df.columns)}"
            )
        self._cache[key] = df
        return df

    def n_to_frame(self, video_id: str, n: int) -> Dict[str, float | int]:
        df = self.load(video_id)
        row = df.loc[df["n"] == int(n)].head(1)
        if row.empty:
            raise KeyError(f"n={n} not found in map for {video_id}")
        r = row.iloc[0]
        return {
            "frame_idx": int(r["frame_idx"]),
            "pts_time": float(r["pts_time"]),
            "fps": float(r["fps"]),
            "n": int(r["n"]),
        }

