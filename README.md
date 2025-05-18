# LLM-FROM-ZERO: Build Your Own Language Model From Scratch

A complete pipeline for training, evaluating, and deploying transformer-based language models  GPT-like from scratch. This project provides a modular and scalable architecture for experimenting with custom language models.

## 🌟 Features

- **Complete Training Pipeline**: End-to-end solution for training transformer models
- **Custom Tokenization**: Adaptive word and character-level tokenization
- **Real-time Monitoring**: Track training progress and model metrics
- **Model Evaluation**: Built-in evaluation metrics including perplexity
- **REST API**: Easy-to-use interface for model inference
- **MinIO Integration**: Efficient model storage and versioning
- **Docker Support**: Containerized deployment for all components

## 🏗️ Architecture

The project consists of several key components:

- **Training Service**: Handles model training and checkpointing
- **Inference Service**: Manages model loading and text generation
- **Model Storage**: MinIO-based artifact storage
- **Database**: PostgreSQL for model metadata and status tracking
- **REST API**: FastAPI-based endpoint for model interaction
- **Monitoring**: Real-time training progress and metrics tracking

### Training Flow
```
                                    Training Pipeline
┌──────────────┐    ┌──────────────┐    ┌──────────────┐    ┌──────────────┐
│              │    │   Training   │    │              │    │              │
│  Input Text  │───>│     Data     │───>│ Tokenization │───>│  Vocabulary  │
│              │    │  Processing  │    │              │    │  Creation    │
└──────────────┘    └──────────────┘    └──────────────┘    └──────────────┘
                                                                    │
┌──────────────┐    ┌──────────────┐    ┌──────────────┐            ▼
│              │    │              │    │              │    ┌──────────────┐
│    Model     │<───│   Training   │<───│    Batch     │<───│  DataLoader  │
│ Checkpointing│    │    Loop      │    │  Generation  │    │              │
└──────────────┘    └──────────────┘    └──────────────┘    └──────────────┘
       │                   │
       │                   │
       ▼                   ▼
┌──────────────┐    ┌──────────────┐
│    MinIO     │    │   Progress   │
│   Storage    │    │  Monitoring  │
└──────────────┘    └──────────────┘
```

### Evaluation Flow
```
                                    Evaluation Pipeline
┌──────────────┐    ┌──────────────┐    ┌──────────────┐    ┌──────────────┐
│              │    │              │    │  Perplexity  │    │              │
│ Load Model   │───>│  Test Data   │───>│ Calculation  │───>│   Metrics    │
│              │    │              │    │              │    │ Aggregation  │
└──────────────┘    └──────────────┘    └──────────────┘    └──────────────┘
       ▲                                                            │
       │                                                            │
┌──────────────┐                                                    ▼
│              │                                           ┌──────────────┐
│    MinIO     │                                           │    JSON      │
│   Storage    │                                           │   Report     │
└──────────────┘                                           └──────────────┘
```

### Inference Flow
```
                                    Inference Pipeline
┌──────────────┐    ┌──────────────┐    ┌──────────────┐    ┌──────────────┐
│              │    │              │    │              │    │              │
│   Prompt     │───>│ Tokenization │───>│   Model      │───>│   Token      │
│              │    │              │    │  Inference   │    │ Generation   │
└──────────────┘    └──────────────┘    └──────────────┘    └──────────────┘
                           ▲                    ▲                    │
                           │                    │                    │
                    ┌──────────────┐    ┌──────────────┐             ▼
                    │              │    │              │    ┌──────────────┐
                    │  Vocabulary  │    │ Load Model   │    │    Text      │
                    │              │    │              │    │  Decoding    │
                    └──────────────┘    └──────────────┘    └──────────────┘
                           ▲                    ▲                    │
                           │                    │                    │
                           └────────────────────┘                    ▼
                    ┌──────────────┐                         ┌──────────────┐
                    │              │                         │              │
                    │    MinIO     │                         │   Response   │
                    │   Storage    │                         │              │
                    └──────────────┘                         └──────────────┘
```

## 🚀 Getting Started

### Prerequisites

- Python 3.10+
- Docker and Docker Compose
- MinIO Client (mc)
- PostgreSQL

### Installation

1. Clone the repository:
```bash
git clone https://github.com/alvaro-brito/llm-from-zero.git
cd llm-by-zero
```

2. Set up the virtual environment:
```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

3. Start the services:
```bash
docker-compose up -d
```

4. Initialize the database and the microservice:
```bash
./start_backend.sh
```

### Training a Model

Use the training script to create and train a new model:

```bash
./train_and_evaluate.sh
```

This will:
1. Create a new model instance
2. Start the training process
3. Monitor training progress
4. Evaluate the model
5. Run inference tests

### Model Evaluation

The system evaluates models using several metrics:
- Perplexity
- Response length statistics
- Processing time
- Sample quality

### Inference

Models can be used for inference through:

1. REST API:
```bash
curl -X POST "http://localhost:8000/api/v1/models/{model_id}/generate" \
     -H "Content-Type: application/json" \
     -d '{"prompt": "Your prompt here", "max_length": 100}'
```

2. Command line:
```bash
./train_and_evaluate.sh --model <model_id>
```

## 🔧 Configuration

Key configuration parameters in `app/core/config.py`:

- `BATCH_SIZE`: Training batch size
- `CONTEXT_LENGTH`: Maximum sequence length
- `D_MODEL`: Model embedding dimension
- `NUM_HEADS`: Number of attention heads
- `NUM_BLOCKS`: Number of transformer blocks
- `LEARNING_RATE`: Training learning rate
- `DROPOUT`: Dropout rate
- `MAX_ITERS`: Maximum training iterations

## 📊 Model Architecture

The transformer model includes:
- Multi-head self-attention
- Position-wise feed-forward networks
- Layer normalization
- Residual connections
- Adaptive tokenization

### Technical Implementation Details

#### 1. Tokenization and Embedding
```
                   Tokenization Pipeline
┌──────────────┐    ┌──────────────┐    ┌──────────────┐
│   Input      │    │  Word-level  │    │  Character   │
│    Text      │───>│ Tokenization │───>│    Level     │
│              │    │              │    │ Tokenization │
└──────────────┘    └──────────────┘    └──────────────┘
                           │                    │
                           └──────────────┐     │
                                          ▼     ▼
                                   ┌──────────────┐
                                   │  Vocabulary  │
                                   │  Creation    │
                                   └──────────────┘
```

- Custom tokenizer combining word and character-level tokenization
- Special tokens handling (`<|endoftext|>`, `<|pad|>`, `<|unk|>`)
- Dynamic vocabulary creation based on frequency

#### 2. Training Process
```
                    Training Architecture
┌──────────────┐    ┌──────────────┐    ┌──────────────┐
│   Token      │    │  Positional  │    │   Input      │
│  Embedding   │───>│  Encoding    │───>│  Embedding   │
└──────────────┘    └──────────────┘    └──────────────┘
        │                                      │
        │           Transformer Block          │
        │           ┌──────────────┐           │
        └──────────>│  Multi-Head  │<──────────┘
                    │  Attention   │
                    └──────────────┘
                           │
                    ┌──────────────┐
                    │    Layer     │
                    │    Norm      │
                    └──────────────┘
                           │
                    ┌──────────────┐
                    │     FFN      │
                    │   Network    │
                    └──────────────┘
                           │
                    ┌──────────────┐
                    │    Layer     │
                    │    Norm      │
                    └──────────────┘
                           │
                           ▼
                    ┌──────────────┐
                    │    Output    │
                    │   Logits     │
                    └──────────────┘
```

Key Components:
- Token Embedding: `nn.Embedding(vocab_size, d_model)`
- Positional Encoding: Sine/cosine based
- Multi-head Attention:
  ```python
  Q = Wq(X)  # [batch_size, seq_len, d_model]
  K = Wk(X)  # [batch_size, seq_len, d_model]
  V = Wv(X)  # [batch_size, seq_len, d_model]
  attention = softmax(QK^T/sqrt(d_k))V
  ```
- Feed Forward Network:
  ```python
  FFN(x) = max(0, xW1 + b1)W2 + b2
  ```

#### 3. Inference Pipeline
```
                    Inference Flow
┌──────────────┐    ┌──────────────┐    ┌──────────────┐
│   Input      │    │  Tokenize &  │    │   Model      │
│   Prompt     │───>│   Embed      │───>│  Forward     │
└──────────────┘    └──────────────┘    └──────────────┘
                                              │
┌──────────────┐    ┌──────────────┐          │
│   Final      │    │   Sample     │          │
│   Text       │<───│   Token      │<─────────┘
└──────────────┘    └──────────────┘
```

Generation Strategy:
- Temperature-based sampling
- Top-p (nucleus) sampling
- Maximum length control
- Special token handling

#### 4. Model Parameters
```
Hyperparameters Configuration
----------------------------
batch_size     = 4      # Training batch size
context_length = 16     # Sequence length
d_model        = 64     # Embedding dimension
num_blocks     = 8      # Transformer blocks
num_heads      = 4      # Attention heads
learning_rate  = 1e-3   # Training rate
dropout        = 0.1    # Dropout rate
max_iters      = 5000   # Training iterations
```

#### 5. Evaluation Metrics
```
                    Evaluation Pipeline
┌──────────────┐    ┌──────────────┐    ┌──────────────┐
│    Model     │    │  Calculate   │    │  Aggregate   │
│  Generation  │───>│  Perplexity  │───>│   Metrics    │
└──────────────┘    └──────────────┘    └──────────────┘
        │                  │                    │
        ▼                  ▼                    ▼
┌──────────────┐    ┌──────────────┐    ┌──────────────┐
│   Response   │    │    Loss      │    │    Final     │
│   Quality    │    │  Analysis    │    │   Report     │
└──────────────┘    └──────────────┘    └──────────────┘
```

Key Metrics:
- Perplexity: exp(average negative log likelihood)
- Response Length Statistics
- Token Distribution Analysis
- Generation Speed

## 🔍 Monitoring

Training progress can be monitored through:
- Real-time progress updates
- Log files in `logs/` directory
- Model status API endpoint

## 📝 API Documentation

The REST API provides endpoints for:

- Model Management:
  - `POST /api/v1/models/create`: Create new model
  - `GET /api/v1/models/{model_id}/status`: Check model status
  - `GET /api/v1/models`: List all models

- Inference:
  - `POST /api/v1/models/{model_id}/generate`: Generate text
  - `POST /api/v1/models/{model_id}/evaluate`: Evaluate model

## 🛠️ Development

### Project Structure

```
llm-by-zero/
├── app/
│   ├── api/           # API endpoints
│   ├── core/          # Core configurations
│   ├── db/            # Database models
│   ├── helpers/       # Utility functions
│   └── service/       # Business logic
├── migrations/        # Database migrations
├── models/           # Saved model checkpoints
├── data/            # Training data
└── docker/          # Docker configurations
```

### Adding New Features

1. Create new endpoints in `app/api/`
2. Implement business logic in `app/service/`
3. Update database models in `app/db/models/`
4. Add migrations using `alembic revision`

## 📈 Performance

Current benchmarks:
- Training time: ~10 minutes for basic model
- Inference latency: ~1.5 seconds per request
- Perplexity: 35-50k range

## 🙏 Acknowledgments

- PyTorch team for the deep learning framework
- Transformer architecture paper authors
- FastAPI for the web framework
- MinIO for object storage

## 📞 Support

For support, please:
1. Check the documentation
2. Search existing issues
3. Open a new issue if needed

---
Built with ❤️ using PyTorch and FastAPI 
