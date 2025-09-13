from __future__ import annotations

from typing import Any, Dict, List, Tuple

from core.logger import SimpleLogger
from core.settings import AppSettings
from service.model_service import ModelService
from .caption_index import get_caption_index


logger = SimpleLogger(__name__)


def _clip_limit_text(text: str, max_tokens: int = 75) -> str:
    if not isinstance(text, str) or not text:
        return text
    parts = text.strip().split()
    if len(parts) <= max_tokens:
        return text.strip()
    return " ".join(parts[:max_tokens]).strip()


def _normalize_for_embedding(text: str) -> str:
    # Keep it simple: prepend neutral descriptor before quoted spans so CLIP sees them
    import re
    if not isinstance(text, str) or not text:
        return text
    out = text
    out = re.sub(r'("[^"]+")', r'phrase \g<0>', out)
    out = re.sub(r'(\u201C[^\u201D]+\u201D)', r'phrase \g<0>', out)
    out = re.sub(r"(?<!\w)('[^']+')", r'phrase \g<0>', out)
    return out


def rrf_fuse(dense: List[Tuple[int, float]], bm25: List[Tuple[int, float]], rrf_k: int = 60, return_k: int = 100) -> List[Tuple[int, float]]:
    # Build rank maps (1-based)
    fused: Dict[int, float] = {}
    for rank, (id_, _) in enumerate(dense, start=1):
        fused[id_] = fused.get(id_, 0.0) + 1.0 / (rrf_k + rank)
    for rank, (id_, _) in enumerate(bm25, start=1):
        fused[id_] = fused.get(id_, 0.0) + 1.0 / (rrf_k + rank)
    # Tie-break by presence in dense (keep stability)
    dense_ids = {id_ for id_, _ in dense}
    items = list(fused.items())
    items.sort(key=lambda x: (x[1], 1 if x[0] in dense_ids else 0), reverse=True)
    return items[: int(return_k)]


def hybrid_rrf(
    query: str,
    model_service: ModelService,
    top_k_dense: int = 200,
    top_k_bm25: int = 200,
    rrf_k: int = 60,
    return_k: int = 100,
) -> List[Dict[str, Any]]:
    app = AppSettings()
    index = get_caption_index()

    # Embed query with existing encoder; respect CLIP context length
    qtext = _clip_limit_text(query)
    qtext = _normalize_for_embedding(qtext)
    try:
        qvec = model_service.embedding(qtext).tolist()[0]
    except Exception:
        # ModelService.embedding returns ndarray; ensure list
        try:
            qvec = model_service.embedding(qtext)[0].tolist()
        except Exception as e:
            logger.warning(f"Dense embedding failed, proceeding sparse-only: {e}")
            qvec = None

    # Dense search via Milvus
    dense: List[Tuple[int, float]] = []
    if qvec is not None:
        dense = index.search_milvus(qvec, top_k=top_k_dense)

    # Sparse search via BM25
    bm25 = index.search_bm25(query, top_k=top_k_bm25)

    # If one side missing, fallback to the other
    if not dense and not bm25:
        return []
    if dense and not bm25:
        fused_ids = [(id_, sc) for id_, sc in dense[: int(return_k)]]
    elif bm25 and not dense:
        fused_ids = [(id_, sc) for id_, sc in bm25[: int(return_k)]]
    else:
        fused_ids = rrf_fuse(dense, bm25, rrf_k=rrf_k, return_k=return_k)

    out: List[Dict[str, Any]] = []
    for id_, fscore in fused_ids:
        meta = index.meta_by_id(int(id_)) or {}
        image_url = index.build_image_url(meta.get('mapped_group'), meta.get('mapped_video'), meta.get('mapped_n'))
        out.append({
            'group': meta.get('group'),
            'video': meta.get('video'),
            'caption': meta.get('caption_text'),
            'score': float(fscore),
            'mapped_n': meta.get('mapped_n'),
            'mapped_frame_idx': meta.get('mapped_frame_idx'),
            'mapped_status': meta.get('mapped_status'),
            'image_url': image_url,
        })
    return out

