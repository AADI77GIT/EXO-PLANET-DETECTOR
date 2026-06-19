from uuid import uuid4

from fastapi import APIRouter, Depends, Header, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import settings
from app.database import get_db
from app.schemas import JobCreateResponse, TrainRequest
from app.services.pipeline import create_job
from celery_worker import train_model_task

train_router = APIRouter(prefix="/api/train", tags=["training"])


def require_api_key(x_api_key: str | None = Header(default=None, alias="X-API-Key")) -> None:
    if x_api_key != settings.api_key:
        raise HTTPException(status_code=403, detail="Forbidden")


@train_router.post("", response_model=JobCreateResponse, dependencies=[Depends(require_api_key)])
async def train(payload: TrainRequest, db: AsyncSession = Depends(get_db)):
    job_id = uuid4().hex
    await create_job(db, job_id, tic_id=0, sector=0)
    train_model_task.apply_async(args=[payload.dataset_path, payload.epochs, payload.batch_size], task_id=job_id)
    return JobCreateResponse(job_id=job_id, status="PENDING", message="Training job queued")
