from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.core.config import settings
from app.core.logging import logger
from app.core.exceptions import ANPRServiceError, anpr_exception_handler
from app.api.router import api_router

from app.services.yolo_filter import get_yolo_model

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info(f"Starting {settings.PROJECT_NAME} v{settings.VERSION}")
    # Pre-load PyTorch YOLO model before any Paddle paddle initialization
    try:
        get_yolo_model()
        logger.info("YOLO v11 model pre-loaded successfully.")
    except Exception as e:
        logger.warning(f"Warning loading YOLO model at startup: {e}")
    yield
    logger.info(f"Shutting down {settings.PROJECT_NAME}")

app = FastAPI(
    title=settings.PROJECT_NAME,
    version=settings.VERSION,
    description="Enterprise Automatic Number Plate Recognition (ANPR) Microservice.",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan
)

if settings.ALLOWED_ORIGINS:
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.ALLOWED_ORIGINS,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

app.add_exception_handler(ANPRServiceError, anpr_exception_handler)

# Include routes directly without versioning prefix
app.include_router(api_router)

@app.get("/", include_in_schema=False)
def root():
    return {
        "service": settings.PROJECT_NAME,
        "version": settings.VERSION,
        "status": "online",
        "docs": "/docs"
    }
