import asyncio
import logging
from typing import List, Dict, Optional
import json
import time
import requests
import tempfile
import os
from sqlalchemy.orm import Session
from app.db.models.llm_model import LLMModel, ModelStatus
from app.service.llm_service import LLMService
from app.core.config import get_settings
from app.helpers.model_trainer import ModelTrainer
from app.helpers.transformer_model import TransformerLanguageModel
from app.db.schemas.llm_schema import LLMModelUpdate
# from evidently.test_suite import TestSuite
# from evidently.tests import TestTextQuality, TestTextDescriptors
import redis
import minio
from minio.error import S3Error
import torch
from urllib.parse import urlparse
from app.core.logging import logger, get_model_logger
from io import BytesIO

settings = get_settings()

class TrainingService:
    def __init__(self, db: Session):
        self.db = db
        self.llm_service = LLMService(db)
        self.model_trainer = ModelTrainer()
        self.redis_client = redis.Redis(
            host=settings.REDIS_HOST,
            port=settings.REDIS_PORT,
            decode_responses=True
        )
        self.minio_client = minio.Minio(
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
                logger.info(f"Created MinIO bucket: {settings.MINIO_BUCKET_NAME}")
        except S3Error as e:
            logger.error(f"Failed to create MinIO bucket: {e}")
            raise

    def _validate_url(self, url: str) -> bool:
        """Validate if the URL is well-formed and accessible."""
        try:
            result = urlparse(url)
            return all([result.scheme, result.netloc])
        except Exception as e:
            logger.error(f"Invalid URL format: {url}")
            return False

    async def _download_training_data(self, urls: List[str]) -> str:
        """Download and combine training data from URLs or local files."""
        try:
            combined_text = []
            for url in urls:
                if url.startswith('file://'):
                    # Handle local file
                    file_path = url.replace('file://', '')
                    try:
                        with open(file_path, 'r', encoding='utf-8') as f:
                            text = f.read().strip()
                            if not text:
                                raise ValueError(f"Empty file: {file_path}")
                            combined_text.append(text)
                    except FileNotFoundError:
                        raise ValueError(f"File not found: {file_path}")
                else:
                    # Handle remote URL
                    if not self._validate_url(url):
                        raise ValueError(f"Invalid URL format: {url}")
                    
                    response = requests.get(url, timeout=30)
                    response.raise_for_status()
                    text = response.text.strip()
                    if not text:
                        raise ValueError(f"Empty response from URL: {url}")
                    combined_text.append(text)
            
            return "\n".join(combined_text)
        except Exception as e:
            logger.error(f"Error processing training data: {str(e)}")
            raise

    def _update_progress(self, model_id: int, progress: float, losses: Optional[Dict[str, float]] = None):
        """Update training progress in database and Redis."""
        model_logger = get_model_logger(model_id)
        try:
            # Update database
            update_data = LLMModelUpdate(progress=float(progress))
            self.llm_service.update_model(model_id, update_data)
            
            # Convert any tensor values to Python floats
            if losses:
                losses = {k: float(v) if isinstance(v, torch.Tensor) else v for k, v in losses.items()}
            
            # Update Redis
            progress_data = {
                "progress": float(progress) if isinstance(progress, torch.Tensor) else progress,
                "losses": losses or {}
            }
            self.redis_client.set(
                f"model:{model_id}:progress",
                json.dumps(progress_data)
            )
            model_logger.debug(f"Progress updated - {progress*100:.2f}% complete")
        except Exception as e:
            model_logger.error(f"Error updating progress: {str(e)}")
            raise

    async def train_model(self, model_id: int):
        """Train a language model with the given data."""
        model_logger = get_model_logger(model_id)
        temp_model_path = None
        start_time = time.time()
        
        try:
            model_logger.info(f"Starting training process for model {model_id}")
            
            # Update model status to training
            self.llm_service.update_model(
                model_id,
                LLMModelUpdate(status=ModelStatus.TRAINING, progress=0.0)
            )
            model_logger.info("Model status updated to TRAINING")

            # Get model data
            model = self.llm_service.get_model(model_id)
            if not model:
                raise ValueError(f"Model {model_id} not found")

            # Download and prepare training data
            model_logger.info("Downloading training data from provided links")
            training_data_links = model.training_data_links
            model_logger.info(f"Training data links: {training_data_links}")
            
            text = await self._download_training_data(training_data_links)
            if not text:
                raise ValueError("No training data available")
            
            # Apply max_samples if specified
            if model.max_samples > 0:
                model_logger.info(f"Limiting dataset to {model.max_samples} samples")
                # Split text into sentences (using periods as delimiter) and take only max_samples
                sentences = [s.strip() + '.' for s in text.split('.') if s.strip()]
                if len(sentences) > model.max_samples:
                    text = ' '.join(sentences[:model.max_samples])
                    model_logger.info(f"Limited training data from {len(sentences)} to {model.max_samples} sentences")
                else:
                    model_logger.info(f"Dataset already has fewer samples ({len(sentences)}) than limit ({model.max_samples})")
            
            model_logger.info(f"Final training data size: {len(text)} characters")

            model_logger.info("Preparing training data")
            train_data, val_data, vocab_size = self.model_trainer.prepare_data(text)
            model_logger.info(f"Data prepared - Vocabulary size: {vocab_size}")

            # Initialize model
            model_logger.info("Initializing transformer model")
            transformer_model = TransformerLanguageModel(
                vocab_size=vocab_size,
                d_model=self.model_trainer.d_model,
                num_heads=self.model_trainer.num_heads,
                num_blocks=self.model_trainer.num_blocks,
                context_length=self.model_trainer.context_length,
                dropout=self.model_trainer.dropout
            ).to(self.model_trainer.device)
            model_logger.info("Model initialized successfully")

            # Train model with progress tracking
            def progress_callback(progress: float, losses: Dict[str, float]):
                self._update_progress(model_id, progress, losses)
                model_logger.info(f"Training progress: {progress*100:.2f}% - Losses: {losses}")

            model_logger.info("Starting model training")
            tracked_losses = self.model_trainer.train_model(
                transformer_model,
                train_data,
                val_data,
                progress_callback
            )
            model_logger.info("Model training completed")

            # Save model
            model_logger.info("Saving model artifacts")
            with tempfile.NamedTemporaryFile(delete=False, suffix='.pt') as tmp:
                temp_model_path = tmp.name
                self.model_trainer.save_model(transformer_model, temp_model_path)
                model_logger.info(f"Model saved to temporary file: {temp_model_path}")

                # Save model to MinIO
                model_path = f"models/{model_id}/model.pt"
                self.minio_client.fput_object(
                    settings.MINIO_BUCKET_NAME,
                    model_path,
                    temp_model_path
                )
                model_logger.info(f"Model uploaded to MinIO: {model_path}")

            # Update model status to completed
            self.llm_service.update_model(
                model_id,
                LLMModelUpdate(
                    status=ModelStatus.COMPLETED,
                    progress=1.0,
                    artifact_path=model_path
                )
            )
            model_logger.info("Model status updated to COMPLETED")

            # Store training metrics
            metrics_path = f"models/{model_id}/metrics.json"
            # Convert tensor values to Python floats
            converted_losses = {}
            for split, losses in tracked_losses.items():
                converted_losses[split] = [float(loss) if isinstance(loss, torch.Tensor) else loss for loss in losses]
            
            metrics_data = json.dumps(converted_losses).encode()
            metrics_buffer = BytesIO(metrics_data)
            self.minio_client.put_object(
                settings.MINIO_BUCKET_NAME,
                metrics_path,
                metrics_buffer,
                len(metrics_data)
            )
            model_logger.info(f"Training metrics saved to: {metrics_path}")

            total_time = time.time() - start_time
            model_logger.info(f"Training completed successfully in {total_time:.2f} seconds")

        except Exception as e:
            model_logger.error(f"Error training model: {str(e)}", exc_info=True)
            self.llm_service.update_model(
                model_id,
                LLMModelUpdate(
                    status=ModelStatus.FAILED,
                    error_message=str(e)
                )
            )
            raise
        finally:
            # Clean up temporary file
            if temp_model_path and os.path.exists(temp_model_path):
                try:
                    os.unlink(temp_model_path)
                    model_logger.info(f"Cleaned up temporary file: {temp_model_path}")
                except Exception as e:
                    model_logger.warning(f"Failed to clean up temporary file {temp_model_path}: {str(e)}")

    async def evaluate_model(self, model_id: int, test_data: List[str]) -> dict:
        """Evaluate a trained model."""
        try:
            model = self.llm_service.get_model(model_id)
            if not model or model.status != ModelStatus.COMPLETED:
                raise ValueError("Model not found or not ready for evaluation")

            # Load model
            model_path = f"models/{model_id}/model.pt"
            self.minio_client.fget_object(
                settings.MINIO_BUCKET_NAME,
                model_path,
                model_path
            )
            
            # Initialize model with same parameters
            transformer_model = TransformerLanguageModel(
                vocab_size=self.model_trainer.encoding.n_vocab,
                d_model=self.model_trainer.d_model,
                num_heads=self.model_trainer.num_heads,
                num_blocks=self.model_trainer.num_blocks,
                context_length=self.model_trainer.context_length,
                dropout=self.model_trainer.dropout
            ).to(self.model_trainer.device)
            
            # Load trained weights
            self.model_trainer.load_model(transformer_model, model_path)

            # Generate predictions
            predictions = []
            for text in test_data:
                # Tokenize input
                tokens = self.model_trainer.encoding.encode(text)
                input_tensor = torch.tensor([tokens], device=self.model_trainer.device)
                
                # Generate response
                output_tokens = transformer_model.generate(input_tensor, max_new_tokens=100)
                output_text = self.model_trainer.encoding.decode(output_tokens[0].tolist())
                predictions.append(output_text)

            # TODO: Implement proper evaluation metrics
            evaluation_results = {
                "predictions": predictions,
                "test_data": test_data
            }
            
            # Store evaluation results
            evaluation_path = f"models/{model_id}/evaluation.json"
            self.minio_client.put_object(
                settings.MINIO_BUCKET_NAME,
                evaluation_path,
                data=json.dumps(evaluation_results).encode(),
                length=len(json.dumps(evaluation_results).encode())
            )

            return evaluation_results

        except Exception as e:
            logger.error(f"Error evaluating model {model_id}: {str(e)}")
            raise 