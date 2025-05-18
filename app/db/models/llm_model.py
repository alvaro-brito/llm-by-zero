from sqlalchemy import Column, Integer, String, Float, DateTime, Enum
from sqlalchemy.sql import func
from app.db.base_class import Base
import enum
import json
from typing import List, Union

class ModelStatus(str, enum.Enum):
    PENDING = "pending"
    TRAINING = "training"
    COMPLETED = "completed"
    FAILED = "failed"

class LLMModel(Base):
    __tablename__ = "llm_models"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, nullable=False)
    description = Column(String, nullable=False)
    _training_data_links = Column('training_data_links', String, nullable=False)  # JSON string of URLs
    max_samples = Column(Integer, nullable=True)
    status = Column(Enum(ModelStatus), default=ModelStatus.PENDING)
    progress = Column(Float, default=0.0)
    artifact_path = Column(String, nullable=True)
    error_message = Column(String, nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())

    @property
    def training_data_links(self) -> List[str]:
        """Get the training data links as a list."""
        return json.loads(self._training_data_links)

    @training_data_links.setter
    def training_data_links(self, value: Union[str, List[str]]):
        """Set the training data links, accepting either a JSON string or a list."""
        if isinstance(value, list):
            self._training_data_links = json.dumps(value)
        elif isinstance(value, str):
            try:
                # Try to parse as JSON first
                json.loads(value)
                self._training_data_links = value
            except json.JSONDecodeError:
                # If not valid JSON, treat as a single URL
                self._training_data_links = json.dumps([value]) 