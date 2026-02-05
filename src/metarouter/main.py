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

    # Test connection to each LM Studio instance with retry
    max_retries = 3
    retry_delay = 2  # seconds
    any_connected = False

    for client in model_router.multi_client.clients:
        connected = False
        delay = retry_delay

        for attempt in range(max_retries):
            try:
                models = await client.get_models(force_refresh=True)
                loaded_models = [m for m in models if m.is_loaded]
                logger.info(
                    f"Connected to LM Studio instance '{client.instance_name}' "
                    f"({client.base_url}) - {len(models)} models, "
                    f"{len(loaded_models)} loaded"
                )
                if loaded_models:
                    logger.info(
                        f"  Loaded: {', '.join(m.id for m in loaded_models)}"
                    )
                connected = True
                any_connected = True
                break

            except Exception as e:
                if attempt < max_retries - 1:
                    logger.warning(
                        f"Failed to connect to '{client.instance_name}' "
                        f"({client.base_url}) attempt {attempt + 1}/{max_retries}: {e}"
                    )
                    logger.info(f"Retrying in {delay} seconds...")
                    await asyncio.sleep(delay)
                    delay *= 2
                else:
                    logger.error(
                        f"Failed to connect to '{client.instance_name}' "
                        f"({client.base_url}) after {max_retries} attempts: {e}"
                    )

        if not connected:
            logger.warning(
                f"Instance '{client.instance_name}' ({client.base_url}) is unreachable - "
                "it can be connected later"
            )

    if any_connected:
        # Check if router model is available on any instance
        all_models = await model_router.multi_client.get_models()
        router_model_loaded = any(
            m.id == settings.router.model and m.is_loaded for m in all_models
        )
        if router_model_loaded:
            logger.info(f"Router model {settings.router.model} is loaded")
        else:
            logger.warning(
                f"Router model {settings.router.model} is NOT loaded on any instance - "
                "please load it in LM Studio for routing to work"
            )
    else:
        logger.error("No LM Studio instances are reachable")
        logger.info("MetaRouter will continue running - instances can be connected later")

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
