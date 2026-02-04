# Quick Start Guide

## Prerequisites Checklist

- [ ] LM Studio installed and running
- [ ] LM Studio API server enabled (Settings → Server → "Start server automatically")
- [ ] phi-4 model downloaded in LM Studio
- [ ] Docker Desktop installed (for Docker method)

## Step-by-Step Setup

### 1. Start LM Studio

1. Open LM Studio
2. Go to Settings → Server
3. Ensure "Start server automatically" is checked
4. Server should be running on `http://localhost:1234`

### 2. Load phi-4 Router Model

1. In LM Studio, search for `microsoft/phi-4`
2. Download if not already downloaded
3. Click "Load Model" - this model must stay loaded for routing to work
4. Verify it's loaded (shows in the top bar)

### 3. Start the Router

#### Option A: Docker (Recommended)

```bash
cd C:\Users\Hunter\Desktop\LLM\metarouter
docker-compose up -d
```

Check it's running:
```bash
docker logs -f metarouter
```

You should see:
```
Connected to LM Studio - X models available
Router model microsoft/phi-4 is loaded ✓
```

#### Option B: Python

```bash
cd C:\Users\Hunter\Desktop\LLM\metarouter

# Create virtual environment
python -m venv .venv
.venv\Scripts\activate

# Install
pip install -e .

# Run
python -m metarouter.main
```

### 4. Test the Router

Run the test script:
```bash
python test_router.py
```

Or test manually:
```bash
curl -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d "{\"messages\": [{\"role\": \"user\", \"content\": \"Say hello\"}]}"
```

### 5. Configure Your Apps

Point your applications to `http://localhost:8000` instead of `http://localhost:1234`

#### Continue.dev

Edit `~/.continue/config.json`:
```json
{
  "models": [{
    "title": "MetaRouter",
    "provider": "openai",
    "model": "gpt-4",
    "apiBase": "http://localhost:8000/v1"
  }]
}
```

#### Python OpenAI Client

```python
from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:8000/v1",
    api_key="sk-dummy"
)

response = client.chat.completions.create(
    model="gpt-4",  # Ignored
    messages=[{"role": "user", "content": "Hello!"}]
)
```

## Verify It's Working

1. Send a code-related query → should route to a coding model
2. Send a simple greeting → should route to a fast small model
3. Check logs to see routing decisions:
   ```bash
   docker logs -f metarouter  # Docker
   # or watch console output if running with Python
   ```

You should see lines like:
```
Selected: qwen3-coder-30b (confidence: 0.95, load_required: False)
Reason: Code query, model already loaded and specialized for programming
```

## Enable Auto-Start on Boot

### Docker Method
1. Ensure Docker Desktop is set to start on login
2. The container will auto-start (already configured with `restart: unless-stopped`)

### Windows Service Method
```bash
winget install NSSM
nssm install MetaRouter "C:\Users\Hunter\Desktop\LLM\metarouter\.venv\Scripts\python.exe"
nssm set MetaRouter AppParameters "-m metarouter.main"
nssm set MetaRouter AppDirectory "C:\Users\Hunter\Desktop\LLM\metarouter"
nssm set MetaRouter Start SERVICE_AUTO_START
nssm start MetaRouter
```

## Common Issues

### "Could not connect to LM Studio"
- Make sure LM Studio is running
- Check API server is enabled in LM Studio settings
- Verify port 1234 is not blocked

### "Router model not loaded"
- Load phi-4 in LM Studio UI
- Router won't work without phi-4 loaded

### Routing to unexpected models
- Check router logs for phi-4's reasoning
- Adjust `prefer_loaded_bonus` in `config/router.yaml`
- Higher value = stronger preference for already-loaded models

## Next Steps

- Install more models in LM Studio (they'll be auto-discovered)
- Watch the logs to understand routing decisions
- Adjust configuration in `config/router.yaml` if needed
- Monitor performance metrics (tracked automatically)

## Getting Help

Check the full README.md for:
- Detailed configuration options
- API documentation
- Troubleshooting guide
- Architecture details
