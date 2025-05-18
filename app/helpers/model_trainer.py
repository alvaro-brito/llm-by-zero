import torch
import torch.nn as nn
from torch.nn import functional as F
import math
import tiktoken
import json
import logging
from typing import List, Dict, Optional, Tuple
from app.core.logging import logger
from app.core.config import get_settings

settings = get_settings()

class ModelTrainer:
    def __init__(self):
        self.batch_size = settings.BATCH_SIZE
        self.context_length = settings.CONTEXT_LENGTH
        self.d_model = settings.D_MODEL
        self.num_blocks = settings.NUM_BLOCKS
        self.num_heads = settings.NUM_HEADS
        self.learning_rate = settings.LEARNING_RATE
        self.dropout = settings.DROPOUT
        self.max_iters = settings.MAX_ITERS
        self.eval_interval = settings.EVAL_INTERVAL
        self.eval_iters = settings.EVAL_ITERS
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        logger.info(f"ModelTrainer initialized with device: {self.device}")
        self.encoding = tiktoken.get_encoding("cl100k_base")

    def prepare_data(self, text: str) -> Tuple[torch.Tensor, torch.Tensor, int]:
        """Prepare data for training with improved tokenization."""
        logger.info("Preparing training data")
        
        # Improved tokenization
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
        vocab = create_vocab(text)
        vocab_size = len(vocab)
        logger.info(f"Vocabulary size: {vocab_size}")
        
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
        tokenizer = CustomTokenizer(vocab)
        
        # Tokenize text
        tokens = tokenizer.encode(text)
        data = torch.tensor(tokens, dtype=torch.long)
        
        # Split into train and validation
        n = int(0.9 * len(data))
        train_data = data[:n].to(self.device)
        val_data = data[n:].to(self.device)
        
        logger.info(f"Train data size: {len(train_data)}, Validation data size: {len(val_data)}")
        return train_data, val_data, vocab_size

    def get_batch(self, split: str, train_data: torch.Tensor, val_data: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Generate a small batch of data of inputs x and targets y."""
        data = train_data if split == 'train' else val_data
        ix = torch.randint(len(data) - self.context_length, (self.batch_size,))
        x = torch.stack([data[i:i+self.context_length] for i in ix])
        y = torch.stack([data[i+1:i+self.context_length+1] for i in ix])
        return x, y

    @torch.no_grad()
    def estimate_loss(self, model: nn.Module, train_data: torch.Tensor, val_data: torch.Tensor) -> Dict[str, float]:
        """Estimate loss on train and validation sets."""
        out = {}
        model.eval()
        for split in ['train', 'valid']:
            losses = torch.zeros(self.eval_iters)
            for k in range(self.eval_iters):
                X, Y = self.get_batch(split, train_data, val_data)
                logits, loss = model(X, Y)
                losses[k] = loss.item()
            out[split] = losses.mean()
        model.train()
        return out

    def train_model(self, model: nn.Module, train_data: torch.Tensor, val_data: torch.Tensor, 
                   progress_callback=None) -> Dict[str, List[float]]:
        """Train the model with progress tracking and learning rate scheduling."""
        try:
            # Initialize optimizer with weight decay
            optimizer = torch.optim.AdamW(model.parameters(), lr=self.learning_rate, weight_decay=0.1)
            
            # Learning rate scheduler
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, 
                T_max=self.max_iters,
                eta_min=self.learning_rate/10
            )
            
            tracked_losses = {'train': [], 'valid': []}
            best_val_loss = float('inf')
            patience = 3
            patience_counter = 0
            
            for step in range(self.max_iters):
                # Evaluate periodically
                if step % self.eval_interval == 0 or step == self.max_iters - 1:
                    losses = self.estimate_loss(model, train_data, val_data)
                    tracked_losses['train'].append(losses['train'])
                    tracked_losses['valid'].append(losses['valid'])
                    
                    # Early stopping check
                    if losses['valid'] < best_val_loss:
                        best_val_loss = losses['valid']
                        patience_counter = 0
                    else:
                        patience_counter += 1
                    
                    if patience_counter >= patience:
                        logger.info("Early stopping triggered")
                        break
                    
                    if progress_callback:
                        progress = (step + 1) / self.max_iters
                        progress_callback(progress, losses)

                # Training step
                xb, yb = self.get_batch('train', train_data, val_data)
                logits, loss = model(xb, yb)
                
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
                scheduler.step()

            return tracked_losses
        except Exception as e:
            logger.error(f"Error training model: {str(e)}")
            raise

    def save_model(self, model: nn.Module, path: str):
        """Save the trained model."""
        try:
            torch.save({
                'model_state_dict': model.state_dict(),
                'hyperparameters': {
                    'batch_size': self.batch_size,
                    'context_length': self.context_length,
                    'd_model': self.d_model,
                    'num_blocks': self.num_blocks,
                    'num_heads': self.num_heads,
                    'learning_rate': self.learning_rate,
                    'dropout': self.dropout
                }
            }, path)
        except Exception as e:
            logger.error(f"Error saving model: {str(e)}")
            raise

    def load_model(self, model: nn.Module, path: str):
        """Load a trained model."""
        try:
            checkpoint = torch.load(path)
            model.load_state_dict(checkpoint['model_state_dict'])
            return model
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            raise 