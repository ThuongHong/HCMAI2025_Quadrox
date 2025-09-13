from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import os
import pickle
import warnings

import pandas as pd

from core.logger import SimpleLogger
from core.settings import AppSettings, KeyFrameIndexMilvusSetting


logger = SimpleLogger(__name__)


@dataclass
class _BM25Pack:
    model: Any
    docs: List[List[str]]
    ids: List[int]


class CaptionIndex:
    def __init__(self, app: AppSettings, milvus_cfg: KeyFrameIndexMilvusSetting):
        self.app = app
        self.milvus_cfg = milvus_cfg
        self.meta_path = Path(app.CAPTION_META_PARQUET)
        self.bm25_path = Path(app.CAPTION_BM25_PKL)
        self.keyframes_root = Path(app.KEYFRAMES_DIR)
        self._meta_df: Optional[pd.DataFrame] = None
        self._bm25: Optional[_BM25Pack] = None
        self._milvus = None
        self._milvus_coll = None

    def _ensure_meta(self):
        if self._meta_df is None:
            if not self.meta_path.exists():
                raise FileNotFoundError(f"Meta parquet not found: {self.meta_path}")
            try:
                self._meta_df = pd.read_parquet(self.meta_path)
            except Exception as e:
                raise RuntimeError(f"Failed to read parquet: {self.meta_path} -> {e}")

    def _ensure_bm25(self):
        if self._bm25 is None:
            if not self.bm25_path.exists():
                raise FileNotFoundError(f"BM25 pickle not found: {self.bm25_path}")
            try:
                pack = pickle.load(open(self.bm25_path, 'rb'))
                self._bm25 = _BM25Pack(model=pack.get('model'), docs=pack.get('docs'), ids=pack.get('ids'))
            except Exception as e:
                raise RuntimeError(f"Failed to load BM25 pickle: {self.bm25_path} -> {e}")

    def _ensure_milvus(self):
        if self._milvus is not None and self._milvus_coll is not None:
            return
        try:
            from pymilvus import connections, Collection
        except Exception as e:
            warnings.warn(f"pymilvus not available: {e}")
            self._milvus = None
            self._milvus_coll = None
            return
        alias = os.environ.get('MILVUS_ALIAS', 'default')
        if connections.has_connection(alias):
            connections.remove_connection(alias)
        params = {
            'host': self.milvus_cfg.HOST,
            'port': self.milvus_cfg.PORT,
            'db_name': os.environ.get('MILVUS_DB', 'default')
        }
        user = os.environ.get('MILVUS_USER')
        pwd = os.environ.get('MILVUS_PASSWORD')
        if user and pwd:
            params['user'] = user
            params['password'] = pwd
        connections.connect(alias=alias, **params)
        try:
            self._milvus_coll = Collection(self.app.CAPTION_MILVUS_COLLECTION, using=alias)
            # Ensure collection is loaded for search to avoid "collection not loaded" errors
            try:
                self._milvus_coll.load()
            except Exception:
                pass
            self._milvus = connections
        except Exception as e:
            warnings.warn(f"Milvus caption collection missing or error: {e}")
            self._milvus = None
            self._milvus_coll = None

    # Public helpers
    def meta_by_id(self, id_: int) -> Optional[Dict[str, Any]]:
        try:
            self._ensure_meta()
            assert self._meta_df is not None
            row = self._meta_df.loc[self._meta_df['id'] == int(id_)].head(1)
            if row.empty:
                return None
            r = row.iloc[0].to_dict()
            return {k: (None if pd.isna(v) else v) for k, v in r.items()}
        except Exception as e:
            logger.warning(f"meta_by_id failed for id={id_}: {e}")
            return None

    def search_bm25(self, query: str, top_k: int = 200) -> List[Tuple[int, float]]:
        self._ensure_bm25()
        assert self._bm25 is not None
        try:
            # Tokenize like during training: use rank_bm25's internal tokenization if provided
            import re
            tokens = re.findall(r"[\w']+", (query or '').lower())
            scores = self._bm25.model.get_scores(tokens)
            # Return top-k (id, score)
            import numpy as np
            idxs = np.argsort(scores)[::-1][:int(top_k)]
            out: List[Tuple[int, float]] = []
            for i in idxs:
                id_ = int(self._bm25.ids[i])
                out.append((id_, float(scores[i])))
            return out
        except Exception as e:
            logger.warning(f"BM25 search failed: {e}")
            return []

    def search_milvus(self, qvec: List[float], top_k: int = 200) -> List[Tuple[int, float]]:
        self._ensure_milvus()
        if self._milvus_coll is None:
            return []
        try:
            # Best-effort load before search
            try:
                self._milvus_coll.load()
            except Exception:
                pass
            res = self._milvus_coll.search(
                data=[qvec],
                anns_field="embedding",
                param=self._milvus_search_params(),
                limit=int(top_k),
                output_fields=["id"],
                _async=False,
            )
            out: List[Tuple[int, float]] = []
            for hits in res:
                for h in hits:
                    out.append((int(h.id), float(h.distance)))
            # distances sorted desc when COSINE
            out.sort(key=lambda x: x[1], reverse=True)
            return out
        except Exception as e:
            # If collection wasn't loaded yet, try one more time after load
            try:
                if "collection not loaded" in str(e).lower():
                    try:
                        self._milvus_coll.load()
                    except Exception:
                        pass
                    res = self._milvus_coll.search(
                        data=[qvec],
                        anns_field="embedding",
                        param=self._milvus_search_params(),
                        limit=int(top_k),
                        output_fields=["id"],
                        _async=False,
                    )
                    out: List[Tuple[int, float]] = []
                    for hits in res:
                        for h in hits:
                            out.append((int(h.id), float(h.distance)))
                    out.sort(key=lambda x: x[1], reverse=True)
                    return out
            except Exception:
                pass
            logger.warning(f"Milvus caption search failed: {e}")
            return []

    def _milvus_search_params(self) -> Dict[str, Any]:
        # Reuse milvus metric from existing settings, with reasonable defaults
        metric = self.milvus_cfg.METRIC_TYPE or 'COSINE'
        index_type = self.milvus_cfg.INDEX_TYPE or 'FLAT'
        params = self.milvus_cfg.SEARCH_PARAMS or {}
        if not params:
            if index_type.upper() == 'IVF_FLAT':
                params = {"metric_type": metric, "params": {"nprobe": 32}}
            else:
                params = {"metric_type": metric, "params": {}}
        return params

    def build_image_url(self, mapped_group: Optional[str], mapped_video: Optional[str], mapped_n: Optional[int]) -> Optional[str]:
        try:
            if not mapped_group or not mapped_video or mapped_n is None:
                return None
            # resources/keyframes/Lxx/Lxx_Vyyy/nnn.jpg
            g = str(mapped_group).replace('K', 'L', 1) if str(mapped_group).startswith('K') else str(mapped_group)
            v = str(mapped_video)
            n = int(mapped_n)
            path = self.keyframes_root / f"{g}" / f"{g}_{v}" / f"{n:03d}.jpg"
            return str(path)
        except Exception:
            return None


@lru_cache(maxsize=1)
def get_caption_index() -> CaptionIndex:
    app = AppSettings()
    milvus_cfg = KeyFrameIndexMilvusSetting()
    return CaptionIndex(app, milvus_cfg)

