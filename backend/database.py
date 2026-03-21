"""Database configuration and ORM models for the Smart Product Categorization System."""
import json
from datetime import datetime
from typing import List, Optional

from sqlalchemy import create_engine, Boolean, DateTime, Float, Integer, String, Text, ForeignKey, text
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, sessionmaker, relationship

DATABASE_URL = "sqlite:///./product_categorization.db"


class Base(DeclarativeBase):
    pass


class PredictionEvent(Base):
    """Records all incoming prediction requests."""
    __tablename__ = "prediction_events"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    timestamp: Mapped[datetime] = mapped_column(DateTime, default=datetime.now)
    predicted_class: Mapped[str] = mapped_column(Text, nullable=False)
    confidence: Mapped[float] = mapped_column(Float, nullable=False)
    latency_ms: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    brightness: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    blur_var: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    width: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    height: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    quality_warnings: Mapped[str] = mapped_column(Text, default="[]")
    image_data_url: Mapped[Optional[str]] = mapped_column(Text, nullable=True)

    feedback: Mapped[Optional["HumanFeedback"]] = relationship(
        "HumanFeedback", back_populates="prediction", uselist=False
    )


class HumanFeedback(Base):
    """Records true labels provided by humans when model confidence is low."""
    __tablename__ = "human_feedback"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    prediction_id: Mapped[int] = mapped_column(
        Integer, ForeignKey("prediction_events.id"), nullable=False
    )
    true_label: Mapped[str] = mapped_column(Text, nullable=False)
    labeled_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.now)

    prediction: Mapped[Optional["PredictionEvent"]] = relationship(
        "PredictionEvent", back_populates="feedback"
    )


class DriftEvent(Base):
    """Records the results of data drift calculations."""
    __tablename__ = "drift_events"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    timestamp: Mapped[datetime] = mapped_column(DateTime, default=datetime.now)
    embedding_score: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    confidence_score: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    class_score: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    is_drift: Mapped[bool] = mapped_column(Boolean, default=False)
    embedding_drifted: Mapped[bool] = mapped_column(Boolean, default=False)
    confidence_drifted: Mapped[bool] = mapped_column(Boolean, default=False)
    class_drifted: Mapped[bool] = mapped_column(Boolean, default=False)


class Alert(Base):
    """Records system alerts for administrators."""
    __tablename__ = "alerts"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    timestamp: Mapped[datetime] = mapped_column(DateTime, default=datetime.now)
    alert_type: Mapped[str] = mapped_column(Text, nullable=False)
    message: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    resolved: Mapped[bool] = mapped_column(Boolean, default=False)


# Database engine and session
engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


def init_db() -> None:
    """Initialize the database by creating all tables."""
    Base.metadata.create_all(bind=engine)

    # Lightweight migration for existing SQLite databases created before
    # image_data_url existed in prediction_events.
    with engine.begin() as conn:
        columns = {
            row[1] for row in conn.execute(text("PRAGMA table_info(prediction_events)"))
        }
        if "image_data_url" not in columns:
            conn.execute(text("ALTER TABLE prediction_events ADD COLUMN image_data_url TEXT"))


def get_db():
    """Dependency to get database session."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def get_quality_warnings(warnings: List[str]) -> str:
    """Serialize quality warnings list to JSON string."""
    return json.dumps(warnings)
