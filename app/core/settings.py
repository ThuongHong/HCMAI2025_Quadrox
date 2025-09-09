from pydantic_settings import BaseSettings
from pydantic import Field
from dotenv import load_dotenv
from pathlib import Path
load_dotenv()

REPO_ROOT = Path(__file__).resolve().parents[2]


class MongoDBSettings(BaseSettings):
    MONGO_HOST: str = Field(..., alias='MONGO_HOST')
    MONGO_PORT: int = Field(..., alias='MONGO_PORT')
    MONGO_DB: str = Field(..., alias='MONGO_DB')
    MONGO_USER: str = Field(..., alias='MONGO_USER')
    MONGO_PASSWORD: str = Field(..., alias='MONGO_PASSWORD')
    MONGO_URI: str = Field(..., alias='MONGO_URI')


class IndexPathSettings(BaseSettings):
    FAISS_INDEX_PATH: str | None
    USEARCH_INDEX_PATH: str | None


class KeyFrameIndexMilvusSetting(BaseSettings):
    COLLECTION_NAME: str = "keyframe_embeddings"
    HOST: str = 'localhost'
    PORT: str = '19530'
    METRIC_TYPE: str = 'COSINE'
    INDEX_TYPE: str = 'FLAT'
    BATCH_SIZE: int = 10000
    SEARCH_PARAMS: dict = {}


class RerankSettings(BaseSettings):
    """Rerank pipeline configuration"""
    # Master switches
    RERANK_ENABLE: bool = Field(
        default=True, description="Master switch for rerank pipeline")
    RERANK_MODE: str = Field(
        default="custom", description="Rerank mode: auto or custom")

    # Sub-switches (SuperGlobal only)
    RERANK_ENABLE_SUPERGLOBAL: bool = Field(
        default=True, description="Enable SuperGlobal rerank")

    # SuperGlobal parameters
    RERANK_SG_TOP_M: int = Field(
        default=200, description="SuperGlobal top-M candidates")
    RERANK_SG_QEXP_K: int = Field(default=5, description="Query expansion K")
    RERANK_SG_IMG_KNN: int = Field(
        default=4, description="Image KNN parameter")
    # New SuperGlobal params (tuned defaults)
    RERANK_SG_ALPHA: float = Field(default=0.85, description="Blend alpha for SG scoring")
    RERANK_SG_BETA: float = Field(default=2.0, description="Weight exponent for neighbor agg")
    RERANK_SG_P_QUERY: float = Field(default=80.0, description="GeM p for query-side pooling")
    # Legacy parameter retained for compatibility
    RERANK_SG_GEM_P: float = Field(
        default=3.0, description="Legacy GeM parameter (compat)")
    RERANK_SG_SCORE_WEIGHT: float = Field(
        default=1.0, description="SuperGlobal score weight")

    # Ensemble and output
    RERANK_FINAL_TOP_K: int = Field(
        default=100, description="Final top-K results")

    # Cache and fallback controls
    RERANK_CACHE_ENABLED: bool = Field(
        default=True, description="Enable caching for rerank stages")
    RERANK_FALLBACK_ENABLED: bool = Field(
        default=True, description="Enable fallback on rerank errors")


class AppSettings(BaseSettings):
    # Model
    MODEL_NAME: str = "ViT-B-32"
    USE_PRETRAINED: bool = True
    PRETRAINED_NAME: str = "openai"

    # Resources
    DATA_FOLDER: str = str(REPO_ROOT / 'resources' / 'keyframes')
    ID2INDEX_PATH: str = str(REPO_ROOT / 'resources' /
                             'keyframes' / 'id2index.json')
    # FRAME2OBJECT: str = str(REPO_ROOT / 'resources' / 'keyframes' / 'detections.json')
    # ASR_PATH: str = str(REPO_ROOT / 'resources' / 'keyframes' / 'asr_proc.json')

    # Query Expansion (QExp) feature flags
    QEXP_ENABLE: bool = Field(default=True)
    QEXP_MAX_VARIANTS: int = Field(default=3)            # take top-N LLM variants
    QEXP_FUSION: str = Field(default="max")              # "max" or "rrf"
    QEXP_OBJECT_FILTER_AUTO: bool = Field(default=True)   # auto-apply object filter if available
