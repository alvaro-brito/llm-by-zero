from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from typing import List, Dict, Any, Optional
from app.core.logging import logger
from app.core.config import get_settings
from app.db.models.llm_model import ModelStatus
from app.service.llm_service import LLMService
from app.helpers.transformer_model import TransformerLanguageModel
from app.helpers.model_trainer import ModelTrainer
import time
import json
from minio.error import S3Error
from io import BytesIO
import tempfile
import os
import requests

settings = get_settings()

class InferenceService:
    def __init__(self, llm_service: LLMService):
        logger.info("Initializing InferenceService")
        self.llm_service = llm_service
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        logger.info(f"Using device: {self.device}")
        self.tokenizer = None
        self.model = None
        self.current_model_id = None
        self.model_trainer = None  # Will be initialized when loading model

    async def load_model(self, model_id: int) -> bool:
        """Load a model for inference."""
        try:
            logger.info(f"Loading model {model_id}")
            
            # Check if model is already loaded
            if self.current_model_id == model_id and self.model is not None:
                logger.info("Model already loaded")
                return True

            # Get model from database
            model = self.llm_service.get_model(model_id)
            if not model:
                logger.error(f"Model {model_id} not found in database")
                return False
            
            if model.status != ModelStatus.COMPLETED:
                logger.error(f"Model {model_id} is not ready. Current status: {model.status}")
                return False

            # Download model from MinIO
            try:
                model_path = f"models/{model_id}/model.pt"
                logger.info(f"Downloading model from MinIO: {model_path}")
                
                with tempfile.NamedTemporaryFile(delete=False, suffix='.pt') as tmp:
                    try:
                        self.llm_service.minio_client.fget_object(
                            settings.MINIO_BUCKET_NAME,
                            model_path,
                            tmp.name
                        )
                        logger.info(f"Model downloaded to temporary file: {tmp.name}")
                        
                        # Load the model
                        checkpoint = torch.load(tmp.name, map_location=self.device)
                        logger.info("Model checkpoint loaded")
                        
                        # Get hyperparameters
                        config = checkpoint.get('hyperparameters', {})
                        logger.info(f"Model config: {config}")

                        # Get vocab size from the model weights
                        vocab_size = checkpoint['model_state_dict']['token_embedding_table.weight'].shape[0]
                        logger.info(f"Model vocabulary size from checkpoint: {vocab_size}")
                        
                        # Initialize our custom transformer model with the same vocab size as the checkpoint
                        self.model = TransformerLanguageModel(
                            vocab_size=vocab_size,  # Use vocab size from checkpoint
                            context_length=config.get('context_length', settings.CONTEXT_LENGTH),
                            d_model=config.get('d_model', settings.D_MODEL),
                            num_heads=config.get('num_heads', settings.NUM_HEADS),
                            num_blocks=config.get('num_blocks', settings.NUM_BLOCKS),
                            dropout=config.get('dropout', settings.DROPOUT)
                        ).to(self.device)
                        
                        logger.info("Model initialized")
                        
                        # Load trained weights
                        self.model.load_state_dict(checkpoint['model_state_dict'])
                        self.model.eval()
                        logger.info("Model weights loaded")

                        # Initialize ModelTrainer and get training data for tokenizer
                        self.model_trainer = ModelTrainer()
                        
                        # Download training data
                        training_data = await self._download_training_data(model.training_data_links[0])
                        if not training_data:
                            raise ValueError("Could not download training data for tokenizer")
                        
                        # Create vocabulary and tokenizer
                        def create_vocab(text):
                            # Add special tokens
                            special_tokens = ['<|endoftext|>', '<|pad|>', '<|unk|>']
                            vocab = {token: i for i, token in enumerate(special_tokens)}
                            
                            # Add word-level tokens
                            words = text.split()
                            word_freq = {}
                            for word in words:
                                if word not in word_freq:
                                    word_freq[word] = 0
                                word_freq[word] += 1
                            
                            # Add most common words to vocabulary
                            common_words = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)
                            for word, freq in common_words:
                                if len(vocab) >= 50000:  # Limit vocab size
                                    break
                                if freq >= 2:  # Only add words that appear at least twice
                                    vocab[word] = len(vocab)
                            
                            # Add character-level tokens for unknown words
                            chars = set(''.join(words))
                            for char in sorted(chars):
                                if char not in vocab:
                                    vocab[char] = len(vocab)
                            
                            return vocab
                        
                        # Create vocabulary
                        vocab = create_vocab(training_data)
                        vocab_size_from_data = len(vocab)
                        
                        if vocab_size_from_data != vocab_size:
                            raise ValueError(f"Vocabulary size mismatch: data has {vocab_size_from_data} tokens, model has {vocab_size} tokens")
                        
                        # Create tokenizer class
                        class CustomTokenizer:
                            def __init__(self, vocab):
                                self.vocab = vocab
                                self.vocab_size = len(vocab)
                                self.inv_vocab = {v: k for k, v in vocab.items()}
                            
                            def encode(self, text):
                                tokens = []
                                words = text.split()
                                for word in words:
                                    if word in self.vocab:
                                        tokens.append(self.vocab[word])
                                    else:
                                        # Handle unknown words character by character
                                        for char in word:
                                            tokens.append(self.vocab.get(char, self.vocab['<|unk|>']))
                                return tokens
                            
                            def decode(self, tokens):
                                return ' '.join(self.inv_vocab.get(token, '<|unk|>') for token in tokens)
                        
                        # Create tokenizer instance
                        self.tokenizer = CustomTokenizer(vocab)
                        logger.info(f"Using custom tokenizer with vocab size: {self.tokenizer.vocab_size}")
                        
                        self.current_model_id = model_id
                        return True
                    
                    except Exception as e:
                        logger.error(f"Error loading model from file: {str(e)}", exc_info=True)
                        return False
                    finally:
                        # Clean up temporary file
                        try:
                            os.unlink(tmp.name)
                            logger.info("Temporary file cleaned up")
                        except Exception as e:
                            logger.warning(f"Failed to clean up temporary file: {str(e)}")
                    
            except S3Error as e:
                logger.error(f"Failed to download model from MinIO: {str(e)}", exc_info=True)
                return False
                
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}", exc_info=True)
            return False

    async def _download_training_data(self, url: str) -> Optional[str]:
        """Download training data from URL or local file."""
        try:
            if url.startswith('file://'):
                # Handle local file
                file_path = url.replace('file://', '')
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        text = f.read().strip()
                        if not text:
                            raise ValueError(f"Empty file: {file_path}")
                        return text
                except FileNotFoundError:
                    raise ValueError(f"File not found: {file_path}")
            else:
                # Handle remote URL
                response = requests.get(url, timeout=30)
                response.raise_for_status()
                text = response.text.strip()
                if not text:
                    raise ValueError(f"Empty response from URL: {url}")
                return text
        except Exception as e:
            logger.error(f"Error downloading training data: {str(e)}")
            return None

    async def generate_response(
        self,
        model_id: int,
        prompt: str,
        max_length: int = 100,
        temperature: float = 0.7,
        top_p: float = 0.9,
        num_return_sequences: int = 1
    ) -> Dict[str, Any]:
        """Generate a response using the loaded model."""
        try:
            logger.info(f"Generating response for model {model_id}")
            logger.info(f"Prompt: {prompt}")
            logger.info(f"Parameters: max_length={max_length}, temperature={temperature}, top_p={top_p}, num_sequences={num_return_sequences}")
            
            start_time = time.time()
            
            # Load model if needed
            if not await self.load_model(model_id):
                error_msg = "Failed to load model"
                logger.error(error_msg)
                return {
                    "error": error_msg,
                    "model_id": model_id
                }

            # Encode input using the custom tokenizer
            logger.info("Encoding input")
            try:
                input_ids = torch.tensor([self.tokenizer.encode(prompt)]).to(self.device)
                logger.info(f"Input shape: {input_ids.shape}")
            except Exception as e:
                error_msg = f"Error encoding input: {str(e)}"
                logger.error(error_msg, exc_info=True)
                return {"error": error_msg}
            
            # Generate
            logger.info("Generating response")
            try:
                with torch.no_grad():
                    # Initialize output with input IDs
                    outputs = input_ids
                    
                    # Generate one token at a time
                    for _ in range(max_length):
                        # Get predictions
                        logits = self.model(outputs)
                        if isinstance(logits, tuple):
                            logits = logits[0]  # Get the main output if it's a tuple
                        
                        # Get next token probabilities for the last position
                        next_token_logits = logits[:, -1, :] / temperature
                        
                        # Apply top-p sampling
                        sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
                        cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
                        sorted_indices_to_remove = cumulative_probs > top_p
                        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                        sorted_indices_to_remove[..., 0] = 0
                        indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
                        next_token_logits[indices_to_remove] = float('-inf')
                        
                        # Sample next token
                        probs = torch.softmax(next_token_logits, dim=-1)
                        next_token = torch.multinomial(probs, num_samples=1)
                        
                        # Append to outputs
                        outputs = torch.cat([outputs, next_token], dim=1)
                        
                        # Check if we should stop (you might want to implement a custom stopping condition)
                        if outputs.size(1) >= max_length:
                            break
            except Exception as e:
                error_msg = f"Error generating response: {str(e)}"
                logger.error(error_msg, exc_info=True)
                return {"error": error_msg}
            
            # Decode output using the custom tokenizer
            logger.info("Decoding response")
            try:
                responses = [
                    self.tokenizer.decode(output[len(input_ids[0]):].tolist())
                    for output in outputs
                ]
                
                processing_time = time.time() - start_time
                return {
                    "responses": responses,
                    "model_id": model_id,
                    "processing_time": processing_time,
                    "metadata": {
                        "input_length": len(input_ids[0]),
                        "output_length": len(outputs[0]),
                        "temperature": temperature,
                        "top_p": top_p
                    }
                }
            except Exception as e:
                error_msg = f"Error decoding response: {str(e)}"
                logger.error(error_msg, exc_info=True)
                return {"error": error_msg}
                
        except Exception as e:
            error_msg = str(e)
            logger.error(f"Inference error: {error_msg}", exc_info=True)
            return {"error": error_msg}

    async def evaluate_model(
        self,
        model_id: int,
        test_data: List[str],
        metrics: List[str] = ["perplexity", "response_length"],
        batch_size: int = 8
    ) -> Dict[str, Any]:
        """Evaluate the model on test data."""
        try:
            start_time = time.time()
            
            # Load model if needed
            if not await self.load_model(model_id):
                return {
                    "error": "Failed to load model",
                    "model_id": model_id
                }

            results = {
                "model_id": model_id,
                "metrics": {},
                "responses": [],
                "processing_time": 0
            }
            
            # Process test data in batches
            for i in range(0, len(test_data), batch_size):
                batch = test_data[i:i + batch_size]
                
                # Generate responses for batch
                for prompt in batch:
                    response = await self.generate_response(model_id, prompt)
                    if "error" not in response:
                        results["responses"].append({
                            "prompt": prompt,
                            "response": response["responses"][0]
                        })
                
            # Calculate metrics
            if len(results["responses"]) > 0:
                if "perplexity" in metrics:
                    total_perplexity = 0
                    for item in results["responses"]:
                        # Tokenize response
                        input_ids = torch.tensor([self.tokenizer.encode(item["response"])]).to(self.device)
                        
                        # Calculate loss
                        with torch.no_grad():
                            logits, loss = self.model(input_ids, input_ids)
                            total_perplexity += torch.exp(loss).item()
                            
                    results["metrics"]["perplexity"] = total_perplexity / len(results["responses"])
                
                if "response_length" in metrics:
                    # Calculate token lengths
                    token_lengths = []
                    for item in results["responses"]:
                        tokens = self.tokenizer.encode(item["response"])
                        token_lengths.append(len(tokens))
                    
                    results["metrics"]["avg_response_length"] = sum(token_lengths) / len(token_lengths)
                    results["metrics"]["min_response_length"] = min(token_lengths)
                    results["metrics"]["max_response_length"] = max(token_lengths)
                
                if "word_length" in metrics:
                    # Calculate word lengths
                    word_lengths = []
                    for item in results["responses"]:
                        words = item["response"].split()
                        word_lengths.append(len(words))
                    
                    results["metrics"]["avg_word_length"] = sum(word_lengths) / len(word_lengths)
                    results["metrics"]["min_word_length"] = min(word_lengths)
                    results["metrics"]["max_word_length"] = max(word_lengths)
            
            results["processing_time"] = time.time() - start_time
            results["total_samples"] = len(test_data)
            results["successful_samples"] = len(results["responses"])
            
            return results
            
        except Exception as e:
            logger.error(f"Error evaluating model: {e}")
            return {
                "error": str(e),
                "model_id": model_id
            } 