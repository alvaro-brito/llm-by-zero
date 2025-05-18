#!/bin/bash

# Exit on error
set -e

echo "🔧 Setting up LLM Training Service..."

# Check if uv is installed
if ! command -v uv &> /dev/null; then
    echo "⚠️  UV not found. Installing UV..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
fi

# Set Python to not create __pycache__ directories
export PYTHONDONTWRITEBYTECODE=1

# Create virtual environment if it doesn't exist
if [ ! -d ".venv" ]; then
    echo "🔨 Creating virtual environment..."
    python3.10 -m venv .venv
fi

# Activate virtual environment
echo "🔌 Activating virtual environment..."
source .venv/bin/activate

# Install dependencies
echo "📦 Installing dependencies..."
uv pip install -e .

# Start infrastructure services
echo "🚀 Starting infrastructure services..."
docker-compose up -d

# Wait for PostgreSQL to be ready
echo "⏳ Waiting for PostgreSQL to be ready..."
until docker exec llm-by-zero-db-1 pg_isready -U postgres; do
    echo "PostgreSQL is unavailable - sleeping"
    sleep 1
done

# Run database migrations
echo "🔄 Running database migrations..."
alembic upgrade head

# Create models directory if it doesn't exist
echo "📁 Creating models directory..."
mkdir -p models

# Start the FastAPI server
echo "🚀 Starting FastAPI server..."
python3 -m uvicorn app.main:app --reload --host 0.0.0.0 --port 8000