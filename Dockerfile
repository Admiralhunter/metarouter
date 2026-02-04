FROM python:3.13-slim

WORKDIR /app

# Copy application code and dependencies
COPY pyproject.toml ./
COPY src ./src
COPY config ./config

# Install dependencies
RUN pip install --no-cache-dir .

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import httpx; httpx.get('http://localhost:8000/health', timeout=5)"

# Run the application
CMD ["python", "-m", "uvicorn", "metarouter.main:app", "--host", "0.0.0.0", "--port", "8000"]
