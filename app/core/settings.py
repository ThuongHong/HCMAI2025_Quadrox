from pydantic_settings import BaseSettings
from pydantic import Field
from dotenv import load_dotenv
load_dotenv()


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
    BATCH_SIZE: int =10000
    SEARCH_PARAMS: dict = {}
    
class AppSettings(BaseSettings):
    # Model
    MODEL_NAME: str = "ViT-B-32"
    USE_PRETRAINED: bool = True
    PRETRAINED_NAME: str = "openai"

    # Resources
    DATA_FOLDER: str  = "/resources/"
    ID2INDEX_PATH: str = "resources/keyframes/id2index.json"
    # FRAME2OBJECT: str = '/media/tinhanhnguyen/Data3/Projects/HCMAI2025_Baseline/app/data/detections.json'
    # ASR_PATH: str = '/media/tinhanhnguyen/Data3/Projects/HCMAI2025_Baseline/app/data/asr_proc.json'


