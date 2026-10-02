"""Service information and health status REST routes."""

from fastapi import APIRouter

from app.core import constants

router = APIRouter(tags=["Info"])


@router.get("/", summary="Service Information")
async def root() -> dict[str, str]:
    """Return microservice name, version, status, and interactive documentation link."""
    return {
        "name": constants.PROJECT_NAME,
        "version": constants.VERSION,
        "status": "running",
        "docs": "/docs",
    }


@router.get("/health", summary="Health Check")
async def health() -> dict[str, str]:
    """Return microservice operational health status confirmation."""
    return {
        "status": "healthy",
        "service": constants.PROJECT_NAME,
        "version": constants.VERSION,
    }
