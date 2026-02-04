"""FastAPI application entry point."""

import asyncio
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

    logger.info(f"LM Studio URL: {settings.lm_studio.base_url}")
    logger.info(f"Router model: {settings.router.model}")

    # Initialize router
    model_router = ModelRouter(settings)
    app.state.router = model_router

    # Test connection to LM Studio with retry
    max_retries = 3
    retry_delay = 2  # seconds
    connected = False

    for attempt in range(max_retries):
        try:
            models = await model_router.client.get_models()
            logger.info(f"Connected to LM Studio - {len(models)} models available")

            loaded_models = [m for m in models if m.is_loaded]
            if loaded_models:
                logger.info(f"Loaded models: {', '.join(m.id for m in loaded_models)}")
            else:
                logger.warning("No models currently loaded in LM Studio")

            # Check if router model is loaded
            router_model_loaded = any(
                m.id == settings.router.model and m.is_loaded for m in models
            )
            if router_model_loaded:
                logger.info(f"Router model {settings.router.model} is loaded")
            else:
                logger.warning(
                    f"Router model {settings.router.model} is NOT loaded - "
                    "please load it in LM Studio for routing to work"
                )
            connected = True
            break

        except Exception as e:
            if attempt < max_retries - 1:
                logger.warning(
                    f"Failed to connect to LM Studio (attempt {attempt + 1}/{max_retries}): {e}"
                )
                logger.info(f"Retrying in {retry_delay} seconds...")
                await asyncio.sleep(retry_delay)
                retry_delay *= 2  # exponential backoff
            else:
                logger.error(f"Failed to connect to LM Studio after {max_retries} attempts: {e}")

    if not connected:
        logger.error(f"Make sure LM Studio is running on {settings.lm_studio.base_url}")
        logger.info("MetaRouter will continue running - LM Studio can be connected later")

    yield

    # Shutdown
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
