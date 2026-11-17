"""FastAPI application entry point."""

import logging
import os
import socket
import sys
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .api.routes import router
from .config.settings import get_settings
from .routing.router import ModelRouter

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)

logger = logging.getLogger(__name__)


def get_local_ip() -> str:
    """Get the local network IP address."""
    try:
        # Connect to an external address to determine the local IP
        # This doesn't actually send any data
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
            s.connect(("8.8.8.8", 80))
            return s.getsockname()[0]
    except Exception:
        return "127.0.0.1"


def is_running_in_docker() -> bool:
    """Check if running inside a Docker container."""
    return os.path.exists("/.dockerenv")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan context manager for startup and shutdown."""
    # Startup
    settings = get_settings()
    logger.info("Starting MetaRouter")

    # Log server access URLs
    local_ip = get_local_ip()
    port = settings.server.port
    logger.info(f"Local:   http://localhost:{port}/")
    if is_running_in_docker():
        logger.info(f"Network: http://<host-ip>:{port}/  (use your host machine's IP)")
    else:
        logger.info(f"Network: http://{local_ip}:{port}/")

    # Log configured LM Studio instances
    instances = settings.lm_studio.get_instances()
    if len(instances) == 1:
        logger.info(f"LM Studio instance: {instances[0].base_url}")
    else:
        logger.info(f"LM Studio instances ({len(instances)}):")
        for inst in instances:
            logger.info(f"  - {inst.name}: {inst.base_url}")
    logger.info(f"Router model: {settings.router.model}")

    # Initialize router
    model_router = ModelRouter(settings)
    app.state.router = model_router

    models = await model_router.client.get_models(force_refresh=True)
    if not models:
        logger.warning("No LM Studio models discovered; discovery will retry")
    elif not any(model.id == settings.router.model and model.is_loaded for model in models):
        logger.warning("Router model %s is not loaded on any instance", settings.router.model)

    model_router.client.start_health_loop()
    try:
        yield
    finally:
        await model_router.client.stop_health_loop()
        logger.info("Shutting down MetaRouter")


# Create FastAPI app
app = FastAPI(
    title="MetaRouter",
    description="LLM-powered intelligent routing for LM Studio",
    version="0.1.0",
    lifespan=lifespan,
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routes
app.include_router(router)


if __name__ == "__main__":
    import uvicorn

    settings = get_settings()

    # Update logging level from settings
    logging.getLogger().setLevel(settings.logging.level)

    uvicorn.run(
        "metarouter.main:app",
        host=settings.server.host,
        port=settings.server.port,
        reload=False,
        log_level=settings.logging.level.lower(),
    )
