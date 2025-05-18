from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.controller.llm_controller import router as llm_router
from app.core.config import get_settings
from app.db.base_class import Base
from app.db.session import engine
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

settings = get_settings()

# Create database tables
Base.metadata.create_all(bind=engine)

app = FastAPI(
    title=settings.PROJECT_NAME,
    description="Microservice for training and managing LLM models from scratch",
    version="0.1.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(llm_router, prefix=settings.API_V1_STR)

@app.get("/")
async def root():
    return {"message": "Welcome to LLM Training Service"}

@app.on_event("startup")
async def startup_event():
    logger.info("Starting up FastAPI application")
    try:
        # Test database connection
        Base.metadata.create_all(bind=engine)
        logger.info("Database connection successful")
    except Exception as e:
        logger.error(f"Error connecting to database: {e}")
        raise

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True) 