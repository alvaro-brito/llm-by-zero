from fastapi import APIRouter, HTTPException, BackgroundTasks, Depends
from sqlalchemy.orm import Session
from app.service.training_service import TrainingService
from app.service.llm_service import LLMService
from app.service.inference_service import InferenceService
from app.db.models.llm_model import ModelStatus
from app.db.schemas.llm_schema import (
    LLMModelCreate,
    LLMModelUpdate,
    LLMModelInDB,
    InferenceRequest,
    InferenceResponse,
    EvaluationRequest,
    EvaluationResponse
)
from app.db.session import get_db
from app.core.logging import logger
import asyncio
from concurrent.futures import ThreadPoolExecutor

router = APIRouter()

def run_training(model_id: int, db: Session):
    """Run the training process in the background."""
    try:
        # Create training service
        training_service = TrainingService(db)
        
        # Create event loop
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        # Run training
        loop.run_until_complete(training_service.train_model(model_id))
        loop.close()
    except Exception as e:
        logger.error(f"Training failed for model {model_id}: {str(e)}")
        # Update model status to failed
        llm_service = LLMService(db)
        llm_service.update_model(
            model_id,
            LLMModelUpdate(
                status=ModelStatus.FAILED,
                error_message=str(e)
            )
        )

@router.post("/models", response_model=LLMModelInDB)
async def generate_model(
    model_data: LLMModelCreate,
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db)
):
    """Generate a new language model."""
    try:
        llm_service = LLMService(db)
        
        max_samples = 0
        if model_data.max_samples:
            max_samples = model_data.max_samples
            
        # Create model
        model = llm_service.create_model({
            "name": model_data.name,
            "description": model_data.description,
            "max_samples": max_samples,
            "training_data_links": list(model_data.training_data_links)
        })
        
        # Start training in background
        background_tasks.add_task(run_training, model.id, db)
        
        logger.info(f"Model {model.id} created and training started")
        return model
    except Exception as e:
        logger.error(f"Error generating model: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/models/{llm_id}", response_model=LLMModelInDB)
async def get_status(
    llm_id: int,
    db: Session = Depends(get_db)
):
    """Get the status of a model."""
    try:
        llm_service = LLMService(db)
        model = llm_service.get_model(llm_id)
        
        if not model:
            raise HTTPException(status_code=404, detail="Model not found")
            
        return model
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting model status: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

def get_inference_service(db: Session = Depends(get_db)) -> InferenceService:
    llm_service = LLMService(db)
    return InferenceService(llm_service)

@router.post("/models/{model_id}/inference", response_model=InferenceResponse)
async def generate_inference(
    model_id: int,
    request: InferenceRequest,
    inference_service: InferenceService = Depends(get_inference_service)
):
    """Generate inference using a trained model."""
    try:
        logger.info(f"Received inference request for model {model_id}")
        response = await inference_service.generate_response(
            model_id=model_id,
            prompt=request.prompt,
            max_length=request.max_length,
            temperature=request.temperature,
            top_p=request.top_p,
            num_return_sequences=request.num_sequences
        )
        
        if "error" in response:
            logger.error(f"Inference error: {response['error']}")
            raise HTTPException(status_code=500, detail=response["error"])
            
        logger.info("Inference completed successfully")
        return InferenceResponse(
            response=response["responses"][0] if response["responses"] else "",
            processing_time=response["processing_time"],
            metadata=response["metadata"]
        )
        
    except Exception as e:
        logger.error(f"Error during inference: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/models/{model_id}/evaluate", response_model=EvaluationResponse)
async def evaluate_model(
    model_id: int,
    request: EvaluationRequest,
    inference_service: InferenceService = Depends(get_inference_service)
):
    """Evaluate a model's performance."""
    try:
        # Extract test prompts from request
        test_prompts = [item["prompt"] for item in request.test_data]
        
        # Call evaluate_model on inference service
        results = await inference_service.evaluate_model(
            model_id=model_id,
            test_data=test_prompts,
            metrics=request.metrics,
            batch_size=request.batch_size
        )
        
        if "error" in results:
            raise HTTPException(status_code=500, detail=results["error"])
            
        return EvaluationResponse(
            results=results["metrics"],
            processing_time=results["processing_time"],
            total_samples=results["total_samples"]
        )
    except Exception as e:
        logger.error(f"Error during evaluation: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e)) 