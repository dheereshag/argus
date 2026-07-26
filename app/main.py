from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.core.config import settings
from app.core.exceptions import ANPRServiceError, anpr_exception_handler
from app.api.router import api_router

@asynccontextmanager
async def lifespan(app: FastAPI):
    print(f"🚀 Starting {settings.PROJECT_NAME} v{settings.VERSION}")
    yield
    print(f"🛑 Shutting down {settings.PROJECT_NAME}")

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
