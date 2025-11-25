import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI
from dotenv import load_dotenv
import uvicorn

from app.api.v1.router import router as api_router
from app.core.dependencies import get_redis
from rag.engine import RAGEngineSingleton

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Manages the lifecycle of the application.
    1. Connects to Redis.
    2. Initializes the RAG Engine (loading models into memory).
    3. Handles shutdown cleanup.
    """
    logger.info("Server starting application...")
    
    # Establish Redis Connection
    redis = await get_redis()
    try:
        await redis.ping()
        logger.info("Redis connection established successfully.")
    except Exception as e:
        logger.error(f"Redis connection failed: {e}")

    # Warm up RAG Engine : This loads the BM25 index and connects to Qdrant before the first request
    try:
        engine = RAGEngineSingleton()
        engine.initialize()
        logger.info("RAG Engine initialized successfully.")
    except Exception as e:
        logger.error(f"RAG Engine initialization failed: {e}")

    yield
    
    # Shutdown Cleanup
    logger.info("Server shutting down...")
    await redis.close()

app = FastAPI(
    title="LF Jobs RAG",
    version="1.0.0",
    lifespan=lifespan
)

app.include_router(api_router, prefix="/api")

if __name__ == "__main__":
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)