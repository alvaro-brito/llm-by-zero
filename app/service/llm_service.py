from sqlalchemy.orm import Session
from app.db.models.llm_model import LLMModel, ModelStatus
from app.db.schemas.llm_schema import LLMModelCreate, LLMModelUpdate, LLMModelInDB
from typing import Optional, Dict, Any, Union
import json
import time
from minio import Minio
from minio.error import S3Error
from app.core.config import get_settings

settings = get_settings()

class LLMService:
    def __init__(self, db: Session):
        self.db = db
        self.minio_client = Minio(
            settings.MINIO_ENDPOINT,
            access_key=settings.MINIO_ACCESS_KEY,
            secret_key=settings.MINIO_SECRET_KEY,
            secure=False
        )
        self._ensure_minio_bucket()

    def _ensure_minio_bucket(self):
        try:
            if not self.minio_client.bucket_exists(settings.MINIO_BUCKET_NAME):
                self.minio_client.make_bucket(settings.MINIO_BUCKET_NAME)
        except S3Error as e:
            raise

    def create_model(self, model_data: dict) -> LLMModelInDB:
        # Ensure training_data_links is a list before JSON serialization
        training_data_links = model_data["training_data_links"]
        if isinstance(training_data_links, str):
            try:
                # Try to parse if it's a JSON string
                training_data_links = json.loads(training_data_links)
            except json.JSONDecodeError:
                # If not JSON, treat as a single URL
                training_data_links = [training_data_links]
        
        db_model = LLMModel(
            name=model_data["name"],
            description=model_data["description"],
            training_data_links=training_data_links,  # The property setter will handle serialization
            max_samples=model_data.get("max_samples"),  # Add max_samples if provided
            status=ModelStatus.PENDING
        )
        self.db.add(db_model)
        self.db.commit()
        self.db.refresh(db_model)
        return LLMModelInDB.from_orm(db_model)

    def get_model(self, model_id: int) -> Optional[LLMModel]:
        return self.db.query(LLMModel).filter(LLMModel.id == model_id).first()

    def update_model(self, model_id: int, update_data: Union[Dict[str, Any], LLMModelUpdate]) -> Optional[LLMModel]:
        db_model = self.get_model(model_id)
        if not db_model:
            return None

        # Convert dictionary to LLMModelUpdate if needed
        if isinstance(update_data, dict):
            update_data = LLMModelUpdate(**{k: v for k, v in update_data.items() if v is not None})

        # Update model fields from the Pydantic model
        update_dict = update_data.model_dump(exclude_unset=True)
        for field, value in update_dict.items():
            if hasattr(db_model, field):
                setattr(db_model, field, value)

        try:
            self.db.commit()
            self.db.refresh(db_model)
        except Exception as e:
            self.db.rollback()
            raise

        return db_model

    def get_model_status(self, model_id: int) -> Optional[dict]:
        db_model = self.get_model(model_id)
        if not db_model:
            return None

        return {
            "id": db_model.id,
            "status": db_model.status,
            "progress": db_model.progress,
            "error_message": db_model.error_message
        }

    async def perform_inference(self, model_id: int, prompt: str) -> Optional[dict]:
        db_model = self.get_model(model_id)
        if not db_model or db_model.status != ModelStatus.COMPLETED:
            return None

        start_time = time.time()
        # TODO: Implement actual inference logic here
        response = "Sample response"  # Placeholder
        processing_time = time.time() - start_time

        return {
            "response": response,
            "model_id": model_id,
            "processing_time": processing_time
        } 