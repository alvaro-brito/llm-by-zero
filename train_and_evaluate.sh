#!/bin/bash

# Exit on any error
set -e

# Parse command line arguments
MODEL_ID=""
while [[ "$#" -gt 0 ]]; do
    case $1 in
        --model) MODEL_ID="$2"; shift ;;
        *) echo "Unknown parameter: $1"; exit 1 ;;
    esac
    shift
done

# Function to run evaluation and inference
run_evaluation_and_inference() {
    local MODEL_ID=$1
    echo "Running evaluation for model $MODEL_ID..."
    EVAL_RESULTS=$(curl -s -X POST "http://localhost:8000/api/v1/models/$MODEL_ID/evaluate" \
        -H "Content-Type: application/json" \
        -d '{
            "test_data": [
                {"prompt": "Machine learning is"},
                {"prompt": "Neural networks can"},
                {"prompt": "Deep learning enables"},
                {"prompt": "Reinforcement learning allows"},
                {"prompt": "Natural language processing helps"}
            ],
            "metrics": ["perplexity", "response_length"],
            "batch_size": 1
        }')
    echo "Evaluation results:"
    echo "$EVAL_RESULTS" | jq '.'

    echo "Running inference..."
    INFERENCE_RESULT=$(curl -s -X POST "http://localhost:8000/api/v1/models/$MODEL_ID/inference" \
        -H "Content-Type: application/json" \
        -d '{
            "prompt": "Machine learning is a field of study that",
            "max_length": 100,
            "temperature": 0.7,
            "top_p": 0.9,
            "num_sequences": 1
        }')
    echo "Inference result:"
    echo "$INFERENCE_RESULT" | jq '.'
}

if [ -n "$MODEL_ID" ]; then
    # Check if model exists and is completed
    echo "Checking status of model $MODEL_ID..."
    STATUS_RESPONSE=$(curl -s "http://localhost:8000/api/v1/models/$MODEL_ID")
    STATUS=$(echo $STATUS_RESPONSE | jq -r '.status')
    
    if [ "$STATUS" = "completed" ]; then
        echo "Model $MODEL_ID is ready for evaluation"
        run_evaluation_and_inference $MODEL_ID
    else
        echo "Model $MODEL_ID is not ready (status: $STATUS)"
        exit 1
    fi
else
    # Create and train new model
    echo "Starting model training and evaluation pipeline..."

    # Create model
    echo "Creating new model..."
    MODEL_RESPONSE=$(curl -s -X POST http://localhost:8000/api/v1/models \
        -H "Content-Type: application/json" \
        -d '{
            "name": "test_model",
            "description": "Test model for ML concepts",
            "training_data_links": ["file://data/sample.txt"],
            "max_samples": 10000
        }')
    MODEL_ID=$(echo $MODEL_RESPONSE | jq -r '.id')
    echo "Created model with ID: $MODEL_ID"

    # Monitor training status
    echo "Monitoring training status..."
    while true; do
        STATUS_RESPONSE=$(curl -s "http://localhost:8000/api/v1/models/$MODEL_ID")
        STATUS_LINE=$(echo $STATUS_RESPONSE | jq -r '"\(.status) - Progress: \(.progress * 100)%"')
        echo "$STATUS_LINE"
        
        STATUS=$(echo $STATUS_RESPONSE | jq -r '.status')
        if [ "$STATUS" = "completed" ]; then
            break
        elif [ "$STATUS" = "failed" ]; then
            ERROR_MSG=$(echo $STATUS_RESPONSE | jq -r '.error_message')
            echo "Model training failed: $ERROR_MSG"
            exit 1
        fi
        sleep 10
    done

    # Run evaluation and inference for the new model
    run_evaluation_and_inference $MODEL_ID
fi

# Print summary
echo "Pipeline completed successfully!"
echo "Model ID: $MODEL_ID"
echo "Check logs for detailed information." 