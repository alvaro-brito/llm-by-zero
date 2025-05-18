from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from datetime import datetime
from ..models.llm_model import ModelStatus

class LLMModelBase(BaseModel):
    name: str
    description: str
    training_data_links: List[str]
    max_samples: Optional[int] = Field(None, description="Maximum number of samples to use for training. If None, use all available data.")

class LLMModelCreate(LLMModelBase):
    pass

class LLMModelUpdate(BaseModel):
    status: Optional[ModelStatus] = None
    progress: Optional[float] = None
    artifact_path: Optional[str] = None
    error_message: Optional[str] = None

class LLMModelInDB(LLMModelBase):
    id: int
    status: ModelStatus
    progress: float
    artifact_path: Optional[str]
    error_message: Optional[str]
    created_at: datetime
    updated_at: datetime

    class Config:
        from_attributes = True

class InferenceRequest(BaseModel):
    prompt: str = Field(..., description="The input text to generate from")
    max_length: Optional[int] = Field(100, description="Maximum length of generated text")
    temperature: Optional[float] = Field(0.7, description="Sampling temperature")
    top_p: Optional[float] = Field(0.9, description="Top-p sampling parameter")
    num_sequences: Optional[int] = Field(1, description="Number of sequences to generate")

class InferenceResponse(BaseModel):
    response: str = Field(..., description="The generated text")
    processing_time: float = Field(..., description="Time taken to generate the response in seconds")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata about the generation process")

class EvaluationRequest(BaseModel):
    test_data: List[Dict[str, str]] = Field(..., description="List of test prompts")
    metrics: List[str] = Field(default_factory=lambda: ["perplexity", "response_length"], description="Metrics to evaluate")
    batch_size: Optional[int] = Field(1, description="Batch size for evaluation")

class EvaluationResponse(BaseModel):
    results: Dict[str, Any] = Field(..., description="Evaluation metrics results")
    processing_time: float = Field(..., description="Time taken for evaluation in seconds")
    total_samples: int = Field(..., description="Number of samples evaluated") 
    