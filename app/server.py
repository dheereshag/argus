"""FastAPI HTTP REST microservice for Argus ANPR."""

from fastapi import FastAPI

from app.api.errors import register_exception_handlers
from app.api.router import api_router
from app.core import constants
from app.core.lifespan import lifespan
from app.core.middleware import register_middleware


def create_app() -> FastAPI:
    """FastAPI application factory configuring lifespan, middleware, and routes."""
    app = FastAPI(
        title=constants.PROJECT_NAME,
        version=constants.VERSION,
        description="Enterprise Automatic Number Plate Recognition (ANPR) Microservice.",
        docs_url=constants.DOCS_URL,
        redoc_url=constants.REDOC_URL,
        lifespan=lifespan,
    )
    register_middleware(app)
    register_exception_handlers(app)
    app.include_router(api_router)
    return app


app = create_app()


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("app.server:app", host=constants.SERVER_HOST, port=constants.SERVER_PORT, reload=True)
