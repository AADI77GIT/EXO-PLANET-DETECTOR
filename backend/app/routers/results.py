from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.responses import FileResponse
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.database import get_db
from app.models import Star
from app.schemas import DetectionResultResponse, PaginatedStarsResponse, PlotType
from app.services.pipeline import latest_detection

results_router = APIRouter(tags=["results"])


@results_router.get("/api/results/{tic_id}", response_model=DetectionResultResponse)
async def get_latest_result(tic_id: int, db: AsyncSession = Depends(get_db)):
    detection = await latest_detection(db, tic_id)
    if detection is None:
        raise HTTPException(status_code=404, detail="No results yet")
    return DetectionResultResponse(
        label=detection.label,
        confidence=detection.confidence,
        period_days=detection.period_days,
        duration_hours=detection.duration_hours,
        depth_ppt=detection.depth_ppt,
        parameter_errors=detection.parameter_errors,
        plots=sorted(detection.plot_paths.keys()),
        created_at=detection.created_at,
    )


@results_router.get("/api/results/{tic_id}/plot/{plot_type}")
async def get_plot(tic_id: int, plot_type: PlotType, db: AsyncSession = Depends(get_db)):
    detection = await latest_detection(db, tic_id)
    if detection is None:
        raise HTTPException(status_code=404, detail="No results yet")
    path = Path(detection.plot_paths.get(plot_type, ""))
    if not path.exists():
        raise HTTPException(status_code=404, detail="Plot file not found")
    return FileResponse(path, media_type="image/png", filename=path.name)


@results_router.get("/api/stars/", response_model=PaginatedStarsResponse)
async def list_stars(page: int = Query(1, ge=1), limit: int = Query(20, ge=1, le=100), db: AsyncSession = Depends(get_db)):
    offset = (page - 1) * limit
    total = await db.scalar(select(func.count()).select_from(Star))
    rows = await db.execute(select(Star).order_by(Star.processed_at.desc()).limit(limit).offset(offset))
    return PaginatedStarsResponse(page=page, limit=limit, total=int(total or 0), items=list(rows.scalars().all()))
