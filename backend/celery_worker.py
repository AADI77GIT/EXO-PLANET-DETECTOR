import asyncio
import logging
import time

from celery import Celery, group

from app.config import settings
from app.database import AsyncSessionLocal
from app.models import Job, JobStatus
from app.services.pipeline import run_full_pipeline, update_job

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

celery_app = Celery("exo_detector", broker=settings.redis_url, backend=settings.redis_url)
celery_app.conf.task_track_started = True
celery_app.conf.worker_prefetch_multiplier = 1


@celery_app.task(name="run_pipeline_task", bind=True, max_retries=2)
def run_pipeline_task(self, tic_id, sector):
    job_id = self.request.id
    started = time.perf_counter()
    logger.info("pipeline task start tic_id=%s sector=%s job_id=%s", tic_id, sector, job_id)

    async def runner():
        async with AsyncSessionLocal() as db:
            await update_job(db, job_id, JobStatus.running)
            try:
                await run_full_pipeline(int(tic_id), int(sector), db, job_id=job_id)
            except Exception as exc:
                job = await db.get(Job, job_id)
                if job and job.status != JobStatus.failed.value:
                    await update_job(db, job_id, JobStatus.failed, error_msg=str(exc))
                raise

    try:
        asyncio.run(runner())
        logger.info("pipeline task end tic_id=%s sector=%s job_id=%s duration_seconds=%.3f", tic_id, sector, job_id, time.perf_counter() - started)
    except Exception as exc:
        logger.exception("pipeline task failed tic_id=%s sector=%s job_id=%s duration_seconds=%.3f", tic_id, sector, job_id, time.perf_counter() - started)
        raise self.retry(exc=exc, countdown=30 * (self.request.retries + 1))


@celery_app.task(name="train_model_task", bind=True, max_retries=0)
def train_model_task(self, dataset_path: str, epochs: int, batch_size: int):
    from app.ml.train import train

    started = time.perf_counter()
    logger.info("training task start job_id=%s dataset=%s", self.request.id, dataset_path)
    train(dataset_path, epochs=epochs, batch_size=batch_size)
    logger.info("training task end job_id=%s duration_seconds=%.3f", self.request.id, time.perf_counter() - started)


def dispatch_sector_group(tic_sector_pairs: list[tuple[int, int]]):
    return group(run_pipeline_task.s(tic_id, sector) for tic_id, sector in tic_sector_pairs).apply_async()
