from datetime import datetime
from enum import Enum

from sqlalchemy import DateTime, Float, ForeignKey, Integer, JSON, String, Text
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.database import Base


class JobStatus(str, Enum):
    pending = "PENDING"
    running = "RUNNING"
    done = "DONE"
    failed = "FAILED"


class DetectionLabel(str, Enum):
    planet = "PLANET"
    eclipsing_binary = "ECLIPSING_BINARY"
    false_positive = "FALSE_POSITIVE"
    starspot = "STARSPOT"


class Star(Base):
    __tablename__ = "stars"

    tic_id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    sector: Mapped[int | None] = mapped_column(Integer)
    fits_path: Mapped[str | None] = mapped_column(Text)
    processed_path: Mapped[str | None] = mapped_column(Text)
    processed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))

    detections: Mapped[list["Detection"]] = relationship(back_populates="star", cascade="all, delete-orphan")


class Detection(Base):
    __tablename__ = "detections"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    tic_id: Mapped[int] = mapped_column(ForeignKey("stars.tic_id", ondelete="CASCADE"), index=True)
    sector: Mapped[int] = mapped_column(Integer)
    label: Mapped[str] = mapped_column(String(32), index=True)
    confidence: Mapped[float] = mapped_column(Float)
    period_days: Mapped[float] = mapped_column(Float)
    duration_hours: Mapped[float] = mapped_column(Float)
    depth_ppt: Mapped[float] = mapped_column(Float)
    parameter_errors: Mapped[dict] = mapped_column(JSON, default=dict)
    plot_paths: Mapped[dict] = mapped_column(JSON, default=dict)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=datetime.utcnow, index=True)

    star: Mapped[Star] = relationship(back_populates="detections")


class Job(Base):
    __tablename__ = "jobs"

    job_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    tic_id: Mapped[int] = mapped_column(Integer, index=True)
    sector: Mapped[int] = mapped_column(Integer)
    status: Mapped[str] = mapped_column(String(16), default=JobStatus.pending.value, index=True)
    result_id: Mapped[int | None] = mapped_column(Integer)
    error_msg: Mapped[str | None] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=datetime.utcnow)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=datetime.utcnow, onupdate=datetime.utcnow)
