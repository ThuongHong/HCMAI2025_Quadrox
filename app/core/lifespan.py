
from contextlib import asynccontextmanager
from fastapi import FastAPI
from motor.motor_asyncio import AsyncIOMotorClient
from beanie import init_beanie

import os
import sys
ROOT_DIR = os.path.abspath(
    os.path.join(
        os.path.dirname(__file__), '../'
    )
)

sys.path.insert(0, ROOT_DIR)


from core.settings import MongoDBSettings, KeyFrameIndexMilvusSetting, AppSettings
from models.keyframe import Keyframe
from factory.factory import ServiceFactory
from core.logger import SimpleLogger
from service.caption_index import get_caption_index
import threading, time

mongo_client: AsyncIOMotorClient = None
service_factory: ServiceFactory = None
logger = SimpleLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    FastAPI lifespan context manager for startup and shutdown events
    """
    logger.info("Starting up application...")
    
    try:
        mongo_settings = MongoDBSettings()
        milvus_settings = KeyFrameIndexMilvusSetting()
        appsetting = AppSettings()
        global mongo_client
        if mongo_settings.MONGO_URI:
            mongo_connection_string = mongo_settings.MONGO_URI
        else:
            mongo_connection_string = (
                f"mongodb://{mongo_settings.MONGO_USER}:{mongo_settings.MONGO_PASSWORD}"
                f"@{mongo_settings.MONGO_HOST}:{mongo_settings.MONGO_PORT}/?authSource=admin"
            )
        
        mongo_client = AsyncIOMotorClient(mongo_connection_string)
        
        await mongo_client.admin.command('ping')
        logger.info("Successfully connected to MongoDB")
        
        database = mongo_client[mongo_settings.MONGO_DB]
        await init_beanie(
            database=database,
            document_models=[Keyframe]
        )
        logger.info("Beanie initialized successfully")
        
        global service_factory
        milvus_search_params = {
            "metric_type": milvus_settings.METRIC_TYPE,
            "params": milvus_settings.SEARCH_PARAMS
        }
        
        service_factory = ServiceFactory(
            milvus_collection_name=milvus_settings.COLLECTION_NAME,
            milvus_host=milvus_settings.HOST,
            milvus_port=milvus_settings.PORT,
            milvus_user="",  
            milvus_password="",  
            milvus_search_params=milvus_search_params,
            model_name=appsetting.MODEL_NAME,
            use_pretrained=appsetting.USE_PRETRAINED,
            pretrained_name=appsetting.PRETRAINED_NAME,
            use_siglip=appsetting.USE_SIGLIP,  # Enable SigLIP
            siglip_model_path=appsetting.SIGLIP_MODEL_PATH,  # SigLIP model path
            mongo_collection=Keyframe
        )
        logger.info("Service factory initialized successfully")
        
        app.state.service_factory = service_factory
        app.state.mongo_client = mongo_client

        logger.info("Application startup completed successfully")

        # Background warmup for Caption Search (beta): load Parquet/BM25 and Milvus collection without blocking
        try:
            if getattr(appsetting, 'CAPTION_SEARCH_ENABLED', False):
                def _warm_caption():
                    t0 = time.time()
                    try:
                        idx = get_caption_index()
                        logger.info("Caption index warmup started…")
                        idx.warmup(load_milvus=True)
                        logger.info(f"Caption index warmup finished in {time.time()-t0:.1f}s")
                    except Exception as e:
                        logger.warning(f"Caption index warmup skipped: {e}")

                threading.Thread(target=_warm_caption, daemon=True).start()
        except Exception:
            pass
        
    except Exception as e:
        logger.error(f"Failed to start application: {e}")
        raise
    
    yield  
    

    logger.info("Shutting down application...")
    
    try:
        if mongo_client:
            mongo_client.close()
            logger.info("MongoDB connection closed")
            
        logger.info("Application shutdown completed successfully")
        
    except Exception as e:
        logger.error(f"Error during shutdown: {e}")

