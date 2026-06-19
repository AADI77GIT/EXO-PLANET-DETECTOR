from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field


class PipelineRunRequest(BaseModel):
    tic_id: int = Field(gt=0)
    sector: int = Field(gt=0)


class TrainRequest(BaseModel):
    dataset_path: str
    epochs: int = Field(default=100, ge=1, le=500)
    batch_size: int = Field(default=64, ge=1, le=4096)


class JobCreateResponse(BaseModel):
    job_id: str
    status: str
    message: str


class JobStatusResponse(BaseModel):
    job_id: str
    status: str
    tic_id: int
    sector: int
    result_id: int | None = None
    error_msg: str | None = None
    created_at: datetime

    model_config = {"from_attributes": True}


class DetectionResultResponse(BaseModel):
    label: str
    confidence: float
    period_days: float
    duration_hours: float
    depth_ppt: float
    parameter_errors: dict
    plots: list[str]
    created_at: datetime


class StarListItem(BaseModel):
    tic_id: int
    sector: int | None = None
    processed_at: datetime | None = None

    model_config = {"from_attributes": True}


class PaginatedStarsResponse(BaseModel):
    page: int
    limit: int
    total: int
    items: list[StarListItem]


class HealthResponse(BaseModel):
    status: str
    db: str
    redis: str
    models_loaded: bool


PlotType = Literal["raw", "denoised", "prob", "phase"]
