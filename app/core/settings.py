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

    # Sub-switches
    RERANK_ENABLE_SUPERGLOBAL: bool = Field(
        default=True, description="Enable SuperGlobal rerank")
    RERANK_ENABLE_CAPTION: bool = Field(
        default=False, description="Enable Caption rerank")
    RERANK_ENABLE_LLM: bool = Field(
        default=False, description="Enable LLM rerank")

    # SuperGlobal parameters
    RERANK_SG_TOP_M: int = Field(
        default=500, description="SuperGlobal top-M candidates")
    RERANK_SG_QEXP_K: int = Field(default=10, description="Query expansion K")
    RERANK_SG_IMG_KNN: int = Field(
        default=10, description="Image KNN parameter")
    RERANK_SG_GEM_P: float = Field(
        default=3.0, description="Generalized mean pooling parameter")
    RERANK_SG_SCORE_WEIGHT: float = Field(
        default=1.0, description="SuperGlobal score weight")

    # Caption parameters
    RERANK_CAPTION_TOP_T: int = Field(
        default=20, description="Caption rerank top-T")
    RERANK_CAPTION_MODEL_NAME: str = Field(
        default="synthetic", description="Caption model name (synthetic, vintern_cpu)")
    RERANK_CAPTION_MAX_NEW_TOKENS: int = Field(
        default=64, description="Caption generation max tokens")
    RERANK_CAPTION_TEMPERATURE: float = Field(
        default=0.0, description="Caption generation temperature")
    RERANK_CAPTION_SCORE_WEIGHT: float = Field(
        default=0.8, description="Caption score weight")
    RERANK_CAPTION_CACHE_DIR: str = Field(
        default="./cache/captions", description="Caption cache directory")

    # New Vintern captioner parameters
    RERANK_CAPTION_VINTERN_MODEL_PATH: str = Field(
        default="./models/Vintern-1B-v3_5", description="Vintern model path")
    RERANK_CAPTION_STYLE: str = Field(
        default="dense", description="Caption style (dense, short, tags, ocr)")
    RERANK_CAPTION_ALLOW_ON_DEMAND: bool = Field(
        default=False, description="Allow on-demand caption generation")
    RERANK_CAPTION_ALPHA: float = Field(
        default=1.0, description="CLIP score weight")
    RERANK_CAPTION_BETA: float = Field(
        default=0.25, description="Caption score weight")
    RERANK_CAPTION_WORKERS: int = Field(
        default=2, description="Max workers for caption generation")

    # Multilingual text embedding
    RERANK_MULTILINGUAL_MODEL_PATH: str = Field(
        default="./models/clip-multilingual/clip-ViT-B-32-multilingual-v1",
        description="Multilingual text embedding model path")

    # LLM parameters
    RERANK_LLM_TOP_T: int = Field(default=5, description="LLM rerank top-T")
    RERANK_LLM_MODEL_NAME: str = Field(
        default="5CD-AI/Vintern-1B-v2", description="LLM model name")
    RERANK_LLM_TIMEOUT_S: int = Field(
        default=15, description="LLM timeout in seconds")
    RERANK_LLM_SCORE_WEIGHT: float = Field(
        default=1.2, description="LLM score weight")
    RERANK_LLM_CACHE_DIR: str = Field(
        default="./cache/llm_scores", description="LLM cache directory")

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
