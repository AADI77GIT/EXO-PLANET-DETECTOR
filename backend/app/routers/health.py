import redis.asyncio as redis
from fastapi import APIRouter
from sqlalchemy import text

from app.config import settings
from app.database import AsyncSessionLocal
from app.ml.cnn_classifier import get_classifier
from app.schemas import HealthResponse

health_router = APIRouter(tags=["health"])


@health_router.get("/api/health", response_model=HealthResponse)
async def health_check():
    db_status = "connected"
    redis_status = "connected"
    try:
        async with AsyncSessionLocal() as db:
            await db.execute(text("SELECT 1"))
    except Exception:
        db_status = "unavailable"
    try:
        client = redis.from_url(settings.redis_url)
        await client.ping()
        await client.aclose()
    except Exception:
        redis_status = "unavailable"
    autoencoder_loaded = (settings.model_dir / "autoencoder_best.pt").exists()
    classifier_loaded = get_classifier().loaded
    return HealthResponse(status="ok", db=db_status, redis=redis_status, models_loaded=bool(autoencoder_loaded and classifier_loaded))
