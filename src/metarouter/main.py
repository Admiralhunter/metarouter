"""FastAPI application entry point."""

import logging
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


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan context manager for startup and shutdown."""
    # Startup
    settings = get_settings()
    logger.info("Starting MetaRouter")
    logger.info(f"LM Studio URL: {settings.lm_studio.base_url}")
    logger.info(f"Router model: {settings.router.model}")

    # Initialize router
    model_router = ModelRouter(settings)
    app.state.router = model_router

    # Test connection to LM Studio
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
            logger.info(f"Router model {settings.router.model} is loaded ✓")
        else:
            logger.warning(
                f"Router model {settings.router.model} is NOT loaded - "
                "please load it in LM Studio for routing to work"
            )

    except Exception as e:
        logger.error(f"Failed to connect to LM Studio: {e}")
        logger.error("Make sure LM Studio is running on http://localhost:1234")

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
