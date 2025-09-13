#!/usr/bin/env python
"""
Index caption JSON files into a local captions index for hybrid search.

Scans resources/captions/**/*.json for items that map a source keyframe name
like "K01_V001_000004.jpg" to a caption text. Builds:

- Parquet metadata: resources/captions_index/captions_meta.parquet
- BM25 pickle:     resources/captions_index/captions_bm25.pkl
- (Optional) Upserts dense vectors into Milvus collection 'caption_text_v1'

Mapping logic for keyframes:
 - Parse source name into group_raw (Kxx), video (Vyyy), frame_hint (int)
 - Normalize group: K -> L (e.g., K01 -> L01)
 - Load CSV map: resources/map-keyframes/{Lxx}_{Vyyy}.csv if exists
 - If CSV exists: find nearest by |frame_idx - frame_hint|
   -> mapped_status = "exact_like"
 - Else: treat frame_hint as ordinal rank among captions within video and
   interpolate to our n -> mapped_status = "approx"
 - If CSV missing or cannot map: mapped_status = "missing"

This script is idempotent and only writes to resources/captions_index/ and
the Milvus collection 'caption_text_v1'. It does not touch existing keyframe
indices or collections.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import warnings
import argparse
import time
from collections import Counter, defaultdict


def _safe_imports():
    # Local imports guarded so the script can still run partial features
    try:
        import pyarrow as pa  # noqa: F401
        import pyarrow.parquet as pq
    except Exception as e:
        raise RuntimeError("pyarrow is required for Parquet output. Please install pyarrow.") from e
    try:
        from rank_bm25 import BM25Okapi
    except Exception as e:
        raise RuntimeError("rank_bm25 is required. Please install rank_bm25.") from e
    return pq, BM25Okapi


ROOT = Path(__file__).resolve().parents[1]
RES_DIR = ROOT / "resources"
CAPTIONS_DIR = RES_DIR / "captions"
# Write index artifacts directly under resources/captions as requested
CAPTIONS_INDEX_DIR = CAPTIONS_DIR
MAP_KEYFRAMES_DIR = RES_DIR / "map-keyframes"
KEYFRAMES_DIR = RES_DIR / "keyframes"

META_PARQUET = CAPTIONS_INDEX_DIR / "captions_meta.parquet"
BM25_PKL = CAPTIONS_INDEX_DIR / "captions_bm25.pkl"
MILVUS_COLLECTION = os.environ.get("CAPTION_MILVUS_COLLECTION", "caption_text_v1")


# ------------- Tiny logger helpers (emoji-first, no color codes) -------------
def _log_info(msg: str):
    print(f"ℹ️  {msg}")

def _log_ok(msg: str):
    print(f"✅ {msg}")

def _log_warn(msg: str):
    print(f"⚠️  {msg}")

def _log_step(msg: str):
    print(f"🔹 {msg}")

def _log_gear(msg: str):
    print(f"⚙️  {msg}")

def _log_upload(msg: str):
    print(f"⬆️  {msg}")


def _parse_source_kf_name(name: str) -> Optional[Tuple[str, str, Optional[int]]]:
    """Parse names like 'K01_V001_000004.jpg' -> (group_raw, video, frame_hint)

    Returns None if parsing fails. frame_hint is int or None.
    """
    try:
        base = Path(name).name
        if "." in base:
            base = base.split(".")[0]
        parts = base.split("_")
        if len(parts) < 3:
            return None
        group_raw = parts[0]  # e.g., K01
        video = parts[1]      # e.g., V001
        frame_str = parts[2]  # e.g., 000004
        frame_hint = int(frame_str)
        return group_raw, video, frame_hint
    except Exception:
        return None


def _normalize_group(group_raw: str) -> str:
    if group_raw.startswith("K"):
        return "L" + group_raw[1:]
    return group_raw


def _load_map_csv(mapped_group: str, video: str):
    import pandas as pd
    # Paths like resources/map-keyframes/L01_V001.csv
    fname = f"{mapped_group}_{video}.csv"
    path = MAP_KEYFRAMES_DIR / fname
    if not path.exists():
        # try alternate prefix
        alt = fname.replace("L", "K", 1) if mapped_group.startswith("L") else fname.replace("K", "L", 1)
        altp = MAP_KEYFRAMES_DIR / alt
        if not altp.exists():
            return None
        path = altp
    try:
        df = pd.read_csv(path)
        req = {"n", "pts_time", "fps", "frame_idx"}
        if not req.issubset(set(df.columns)):
            warnings.warn(f"CSV schema mismatch for {path.name}; expected {sorted(req)}")
            return None
        return df
    except Exception:
        return None


def _build_records(captions_by_video: Dict[Tuple[str, str], List[Dict[str, Any]]]) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    per_key_counts: Dict[Tuple[str, str], int] = defaultdict(int)
    per_key_status: Dict[Tuple[str, str], Counter] = defaultdict(Counter)
    for (group_raw, video), items in captions_by_video.items():
        mapped_group = _normalize_group(group_raw)
        mapped_video = video
        df = _load_map_csv(mapped_group, mapped_video)
        # Build mapping helpers
        df_sorted = None
        if df is not None:
            df_sorted = df.sort_values(by=["frame_idx"]).reset_index(drop=True)
            _log_step(f"Building: {mapped_group}_{mapped_video} | captions={len(items)} | CSV=✅")
        else:
            _log_step(f"Building: {mapped_group}_{mapped_video} | captions={len(items)} | CSV=⚠️ missing")
        total = max(1, len(items))
        for idx, it in enumerate(items):
            cap_text = it.get("caption") or it.get("text") or it.get("caption_text") or ""
            source_name = it.get("name") or it.get("file") or it.get("image") or it.get("source") or it.get("keyframe")
            if not source_name:
                # Build from known pattern
                # If provided structure contains just frame index, we cannot infer
                source_name = f"{group_raw}_{video}_{it.get('frame', it.get('n','000000'))}.jpg"
            parsed = _parse_source_kf_name(str(source_name))
            frame_hint = parsed[2] if parsed else None

            mapped_n: Optional[int] = None
            mapped_frame_idx: Optional[int] = None
            mapped_status = "missing"

            if df is not None and frame_hint is not None:
                # Interpret hint as frame_idx-like; nearest match
                try:
                    # |frame_idx - frame_hint| minimal
                    df["_diff"] = (df["frame_idx"] - int(frame_hint)).abs()
                    row = df.sort_values("_diff").iloc[0]
                    try:
                        mapped_n = int(row["n"])  # type: ignore[index]
                    except Exception:
                        mapped_n = None
                    try:
                        mapped_frame_idx = int(row["frame_idx"])  # type: ignore[index]
                    except Exception:
                        mapped_frame_idx = None
                    mapped_status = "exact_like"
                except Exception:
                    mapped_status = "missing"
            elif df is not None and frame_hint is None:
                # Approximate by ordinal ratio if possible
                try:
                    pos = idx / max(1, total - 1)
                    n_min = int(df_sorted["n"].min())  # type: ignore[index]
                    n_max = int(df_sorted["n"].max())  # type: ignore[index]
                    mapped_n = int(round(n_min + pos * (n_max - n_min)))
                    row = df.loc[df["n"] == mapped_n]
                    if not row.empty:
                        mapped_frame_idx = int(row.iloc[0]["frame_idx"])  # type: ignore[index]
                    mapped_status = "approx"
                except Exception:
                    mapped_status = "missing"
            else:
                # CSV missing
                mapped_status = "missing"

            rec = {
                "id": None,  # fill later
                "group": group_raw,
                "video": video,
                "source_kf_name": str(source_name),
                "caption_text": str(cap_text),
                "frame_hint": int(frame_hint) if frame_hint is not None else None,
                "mapped_group": mapped_group,
                "mapped_video": mapped_video,
                "mapped_n": mapped_n,
                "mapped_frame_idx": mapped_frame_idx,
                "mapped_status": mapped_status,
            }
            records.append(rec)
            per_key_counts[(group_raw, video)] += 1
            per_key_status[(group_raw, video)][mapped_status] += 1
    # Assign stable integer IDs
    for i, r in enumerate(records):
        r["id"] = i
    # Quick summary per first few videos (debug aid)
    if records:
        _log_step("Per-video caption counts (first 5):")
        shown = 0
        for (g, v), c in per_key_counts.items():
            _log_info(f"  • {g}_{v}: {c} captions")
            st = per_key_status.get((g, v), Counter())
            if st:
                _log_info(f"    ↳ exact:{st.get('exact_like',0)} approx:{st.get('approx',0)} missing:{st.get('missing',0)}")
            shown += 1
            if shown >= 5:
                break
    return records


def _tokenize_for_bm25(texts: List[str]) -> List[List[str]]:
    import re
    docs: List[List[str]] = []
    for t in texts:
        if not isinstance(t, str):
            t = str(t)
        # basic tokenization; lowercase
        tokens = re.findall(r"[\w']+", t.lower())
        docs.append(tokens)
    return docs


def _save_parquet(records: List[Dict[str, Any]]):
    pq, _ = _safe_imports()
    import pyarrow as pa
    CAPTIONS_INDEX_DIR.mkdir(parents=True, exist_ok=True)
    # Convert list of dicts to Arrow Table
    # Normalize types
    def _norm(v):
        return v
    cols = {
        "id": [int(r.get("id", 0)) for r in records],
        "group": [str(r.get("group", "")) for r in records],
        "video": [str(r.get("video", "")) for r in records],
        "source_kf_name": [str(r.get("source_kf_name", "")) for r in records],
        "caption_text": [str(r.get("caption_text", "")) for r in records],
        "frame_hint": [r.get("frame_hint") for r in records],
        "mapped_group": [str(r.get("mapped_group", "")) for r in records],
        "mapped_video": [str(r.get("mapped_video", "")) for r in records],
        "mapped_n": [r.get("mapped_n") for r in records],
        "mapped_frame_idx": [r.get("mapped_frame_idx") for r in records],
        "mapped_status": [str(r.get("mapped_status", "")) for r in records],
    }
    table = pa.Table.from_pydict(cols)
    pq.write_table(table, META_PARQUET)
    _log_ok(f"Parquet saved → {META_PARQUET} (rows={len(records)})")


def _build_bm25(records: List[Dict[str, Any]]):
    _, BM25Okapi = _safe_imports()
    CAPTIONS_INDEX_DIR.mkdir(parents=True, exist_ok=True)
    import pickle
    texts = [r.get("caption_text", "") for r in records]
    docs = _tokenize_for_bm25(texts)
    bm25 = BM25Okapi(docs)
    payload = {
        "model": bm25,
        "docs": docs,
        "ids": [int(r["id"]) for r in records],
    }
    with open(BM25_PKL, "wb") as f:
        pickle.dump(payload, f)
    _log_ok(f"BM25 saved → {BM25_PKL} (docs={len(docs)})")


def _connect_milvus():
    _log_step("_connect_milvus() start")
    try:
        from pymilvus import connections, Collection, FieldSchema, CollectionSchema, DataType, utility
    except Exception as e:
        warnings.warn(f"pymilvus missing or failed to import: {e}; skipping dense upsert")
        return None

    host = os.environ.get("MILVUS_HOST", "localhost")
    port = os.environ.get("MILVUS_PORT", "19530")
    alias = os.environ.get("MILVUS_ALIAS", "default")
    db_name = os.environ.get("MILVUS_DB", "default")
    try:
        if connections.has_connection(alias):
            connections.remove_connection(alias)
        connections.connect(alias=alias, host=host, port=port, db_name=db_name)
        _log_ok(f"Milvus connected ({host}:{port}, alias={alias}, db={db_name})")
        return {
            "connections": connections,
            "Collection": Collection,
            "FieldSchema": FieldSchema,
            "CollectionSchema": CollectionSchema,
            "DataType": DataType,
            "utility": utility,
            "alias": alias,
        }
    except Exception as e:
        warnings.warn(f"Milvus connect failed: {e}; skipping dense upsert")
        return None


def _ensure_caption_collection(milvus, dim: int, metric: str = "COSINE"):
    _log_step(f"_ensure_caption_collection(dim={dim}, metric={metric})")
    Collection = milvus["Collection"]
    FieldSchema = milvus["FieldSchema"]
    CollectionSchema = milvus["CollectionSchema"]
    DataType = milvus["DataType"]
    utility = milvus["utility"]
    alias = milvus["alias"]

    if utility.has_collection(MILVUS_COLLECTION, using=alias):
        _log_info(f"Collection exists: {MILVUS_COLLECTION}")
        return Collection(MILVUS_COLLECTION, using=alias)

    fields = [
        FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=False),
        FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=dim),
    ]
    schema = CollectionSchema(fields, description="Caption text embeddings")
    coll = Collection(name=MILVUS_COLLECTION, schema=schema, using=alias)
    try:
        index_params = {"index_type": "IVF_FLAT", "metric_type": metric, "params": {"nlist": 1024}}
        coll.create_index(field_name="embedding", index_params=index_params)
        _log_ok(f"Index created on 'embedding' ({index_params})")
    except Exception as e:
        _log_warn(f"Index creation skipped/failed: {e}")
    return coll


def _load_text_encoder():
    """Load the same text encoder as the app (open_clip)."""
    _log_step("_load_text_encoder() start")
    try:
        import open_clip
        import torch
        import numpy as np
        model_name = os.environ.get("MODEL_NAME", "ViT-B-32")
        pretrained = os.environ.get("PRETRAINED_NAME", "openai")
        model, _, preprocess = open_clip.create_model_and_transforms(model_name, pretrained=pretrained, force_quick_gelu=True)
        tokenizer = open_clip.get_tokenizer(model_name)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = model.to(device)
        model.eval()
        def embed(text: str) -> List[float]:
            # Respect CLIP ~77 token context; approx 75 tokens
            parts = (text or "").strip().split()
            if len(parts) > 75:
                text = " ".join(parts[:75])
            tokens = tokenizer([text]).to(device)
            with torch.no_grad():
                vec = model.encode_text(tokens).cpu().detach().numpy().astype("float32")[0]
            return vec.tolist()
        # infer dim by encoding a dummy token
        dim = len(embed("hello world"))
        _log_ok(f"Text encoder ready (model={model_name}/{pretrained}, dim={dim})")
        return embed, dim
    except Exception as e:
        warnings.warn(f"Dense embedding disabled (encoder load failed): {e}")
        return None, None


def _upsert_dense_vectors(records: List[Dict[str, Any]]):
    embed, dim = _load_text_encoder()
    if embed is None or dim is None:
        return
    milvus = _connect_milvus()
    if milvus is None:
        return
    coll = _ensure_caption_collection(milvus, dim=dim, metric=os.environ.get("MILVUS_METRIC", "COSINE"))
    # Prepare data
    ids = [int(r["id"]) for r in records]
    vecs = [embed(r.get("caption_text", "")) for r in records]
    try:
        _log_upload(f"Upserting {len(ids)} vectors into '{MILVUS_COLLECTION}' …")
        coll.insert([ids, vecs])
        coll.flush()
        _log_ok(f"Milvus upsert completed (count={len(ids)})")
    except Exception as e:
        warnings.warn(f"Milvus upsert failed: {e}")


def _parse_video_from_json_path(p: Path) -> Tuple[Optional[str], Optional[str]]:
    """Infer (group, video) from filename like L03_V001.json under L03/.
    Returns (group, video) as strings (e.g., 'L03', 'V001') or (None, None).
    """
    try:
        stem = p.stem  # e.g., L03_V001
        parts = stem.split('_')
        if len(parts) >= 2 and (parts[0].startswith(('L', 'K')) and parts[1].startswith('V')):
            return parts[0], parts[1]
    except Exception:
        pass
    # Try parent directory name as group
    try:
        parent = p.parent.name  # e.g., L03
        if parent.startswith(('L', 'K')):
            return parent, None
    except Exception:
        pass
    return None, None


def _load_caption_jsons() -> Dict[Tuple[str, str], List[Dict[str, Any]]]:
    """Return grouped captions by (group_raw, video).

    Expects files under resources/captions/**.json with either:
     - list of objects having fields {name, caption}, or
     - dict mapping name -> caption
    """
    out: Dict[Tuple[str, str], List[Dict[str, Any]]] = {}
    if not CAPTIONS_DIR.exists():
        return out
    files = list(CAPTIONS_DIR.rglob("*.json"))
    _log_step(f"Scanning captions under: {CAPTIONS_DIR}")
    _log_info(f"Found {len(files)} JSON files")
    for p in files:
        fb_group, fb_video = _parse_video_from_json_path(p)
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            _log_warn(f"Failed to parse JSON: {p}")
            continue
        items: List[Tuple[str, str]] = []
        if isinstance(data, dict):
            for k, v in data.items():
                if isinstance(v, str):
                    items.append((k, v))
                elif isinstance(v, dict):
                    cap = v.get("caption") or v.get("text") or ""
                    items.append((k, cap))
        elif isinstance(data, list):
            for obj in data:
                if isinstance(obj, dict):
                    name = obj.get("name") or obj.get("file") or obj.get("image") or obj.get("keyframe") or obj.get("source")
                    cap = obj.get("caption") or obj.get("text") or obj.get("caption_text") or ""
                    if name:
                        items.append((name, cap))
                    else:
                        # Fallback: if keyframe missing but we have group/video from path and maybe 'n'/'frame'
                        n_hint = obj.get("n") or obj.get("frame") or obj.get("idx")
                        if (fb_group and fb_video) and isinstance(n_hint, int):
                            items.append((f"{fb_group}_{fb_video}_{int(n_hint):06d}.jpg", cap))
        # Group by parsed (group_raw, video)
        for name, cap in items:
            parsed = _parse_source_kf_name(str(name))
            if not parsed:
                # Try fallback from path
                if fb_group is None or fb_video is None:
                    continue
                group_raw, video, frame_hint = fb_group, fb_video, None
            else:
                group_raw, video, frame_hint = parsed
            key = (group_raw, video)
            out.setdefault(key, []).append({
                "name": name,
                "caption": cap,
                "frame_hint": frame_hint,
            })
    # Robust to missing groups: just return what we have
    return out


def main():
    parser = argparse.ArgumentParser(description="Index external caption JSONs for hybrid search (BM25 + dense)")
    parser.add_argument("--no-dense", action="store_true", help="Skip dense embedding upsert to Milvus")
    parser.add_argument("--debug", action="store_true", help="Show extra debug logs")
    args = parser.parse_args()

    pq, _ = _safe_imports()
    CAPTIONS_INDEX_DIR.mkdir(parents=True, exist_ok=True)

    _log_gear(f"Repo root: {ROOT}")
    _log_gear(f"Captions dir: {CAPTIONS_DIR}")
    _log_gear(f"Index dir: {CAPTIONS_INDEX_DIR}")
    _log_gear(f"MAP CSV dir: {MAP_KEYFRAMES_DIR}")
    _log_gear(f"Milvus collection: {MILVUS_COLLECTION}")

    t0 = time.time()
    groups = _load_caption_jsons()
    if not groups:
        _log_warn("No captions found; aborting.")
        return
    _log_info(f"Grouped into {len(groups)} (group,video) buckets")
    records = _build_records(groups)

    # Mapping status summary
    status_ctr = Counter([r.get("mapped_status") for r in records])
    _log_step("Mapping status summary:")
    for k in ("exact_like", "approx", "missing"):
        _log_info(f"  • {k}: {status_ctr.get(k, 0)}")

    _save_parquet(records)
    _build_bm25(records)

    # Optional dense upsert
    env_dense = os.environ.get("CAPTION_DENSE_UPSERT", "1").strip().lower() not in ("0", "false")
    do_dense = (not args.no_dense) and env_dense
    if do_dense:
        _upsert_dense_vectors(records)
    else:
        _log_warn("Dense upsert skipped (use --no-dense or CAPTION_DENSE_UPSERT=0 to control)")

    dt = time.time() - t0
    _log_ok(f"Indexed {len(records)} captions in {dt:.1f}s")
    _log_info(f"Parquet → {META_PARQUET}")
    _log_info(f"BM25 → {BM25_PKL}")
    _log_info(f"Milvus → {MILVUS_COLLECTION} (if enabled)")


if __name__ == "__main__":
    main()
