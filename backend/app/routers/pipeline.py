from uuid import uuid4

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.database import get_db
from app.models import Job
from app.schemas import JobCreateResponse, JobStatusResponse, PipelineRunRequest
from app.services.pipeline import create_job
from celery_worker import run_pipeline_task

pipeline_router = APIRouter(prefix="/api/pipeline", tags=["pipeline"])
ACTIVE_STATUSES = {"PENDING", "RUNNING"}


@pipeline_router.post("/run", response_model=JobCreateResponse)
async def run_pipeline(payload: PipelineRunRequest, db: AsyncSession = Depends(get_db)):
    existing = await db.execute(
        select(Job)
        .where(Job.tic_id == payload.tic_id, Job.sector == payload.sector, Job.status.in_(ACTIVE_STATUSES))
        .order_by(Job.created_at.desc())
    )
    active = existing.scalars().first()
    if active is not None:
        return JobCreateResponse(job_id=active.job_id, status=active.status, message="Existing active pipeline job returned")
    job_id = uuid4().hex
    await create_job(db, job_id, payload.tic_id, payload.sector)
    run_pipeline_task.apply_async(args=[payload.tic_id, payload.sector], task_id=job_id)
    return JobCreateResponse(job_id=job_id, status="PENDING", message="Pipeline job queued")


@pipeline_router.get("/status/{job_id}", response_model=JobStatusResponse)
async def pipeline_status(job_id: str, db: AsyncSession = Depends(get_db)):
    job = await db.get(Job, job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Job not found")
    return job
