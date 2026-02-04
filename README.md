# MetaRouter

**LLM-powered intelligent routing for LM Studio**

MetaRouter uses an LLM (phi-4) to automatically select the best model for each query—no training data required, zero configuration needed.

## Features

- **LLM-Based Routing**: Uses phi-4 to understand queries semantically and select optimal models
- **Zero Configuration**: Automatically discovers models from LM Studio API—no manual setup
- **Performance Learning**: Tracks real inference metrics and adapts to your hardware
- **Latency-Aware**: Strongly prefers already-loaded models to minimize response time
- **OpenAI Compatible**: Drop-in replacement for OpenAI API
- **Streaming Support**: Full SSE streaming passthrough
- **Explainable**: Router provides reasoning for each model selection decision
- **Zero Maintenance**: Install new models in LM Studio → automatically available

## Why MetaRouter?

Traditional routing approaches require training data, classifiers, or manual configuration. MetaRouter takes a novel approach:

**Instead of training a classifier to route to models, we use an LLM to route to models.**

### Advantages

✅ **No Training Required** - phi-4 already understands model capabilities semantically
✅ **Adapts to New Models** - Works with any model you install without retraining
✅ **Explainable Decisions** - phi-4 provides reasoning for each routing choice
✅ **Semantic Understanding** - Routes based on query meaning, not just keywords
✅ **Self-Learning** - Incorporates real performance metrics into routing decisions

## How MetaRouter Compares

| Feature | MetaRouter | RouteLLM | Olla | LiteLLM |
|---------|-----------|----------|------|---------|
| Routing Method | LLM-based | ML Classifiers | Load Balancing | Cost/Fallback |
| Training Required | No | Yes | No | No |
| Local LLM Focus | Yes | Partial | Yes | No |
| Auto Model Discovery | Yes | No | Partial | No |
| Performance Learning | Yes | No | No | No |
| LM Studio Specific | Yes | No | Yes | No |
| Explainable Routing | Yes | No | No | No |

**When to use MetaRouter:**
- You run multiple local models with different capabilities (code, vision, reasoning, chat)
- You want intelligent routing without training classifiers
- You need explainable routing decisions
- You want zero-configuration auto-discovery

**Related Projects:**
- [RouteLLM](https://github.com/lm-sys/RouteLLM) - ML-based routing framework (requires training data)
- [Olla](https://github.com/thushan/olla) - High-performance LLM proxy (load balancing focus)
- [LiteLLM](https://github.com/BerriAI/litellm) - Multi-provider orchestration (cloud API focus)

## How It Works

```
Client Request → Router → phi-4 selects best model → Forward to LM Studio → Response
```

1. Router queries LM Studio for available models and their load states
2. phi-4 receives the model list with capabilities and performance data
3. phi-4 selects the best model considering: task type, quality, speed, and load state
4. Request is forwarded to the selected model via LM Studio

## Prerequisites

- **LM Studio** running on `http://localhost:1234`
- **phi-4 model** loaded in LM Studio (used for routing decisions)
- **Python 3.12+** (if running without Docker)
- **Docker Desktop** (recommended for isolation)

## Quick Start

### Option 1: Docker (Recommended)

1. **Start LM Studio and load phi-4**:
   ```bash
   # Make sure LM Studio is running on port 1234
   # Load microsoft/phi-4 in LM Studio UI
   ```

2. **Build and start the router**:
   ```bash
   cd metarouter
   docker-compose up -d
   ```

3. **Check logs**:
   ```bash
   docker logs -f metarouter
   ```

4. **Test the router**:
   ```bash
   curl -X POST http://localhost:8000/v1/chat/completions \
     -H "Content-Type: application/json" \
     -d '{
       "messages": [{"role": "user", "content": "Write a Python function"}]
     }'
   ```

### Option 2: Python Virtual Environment

1. **Create and activate venv**:
   ```bash
   cd metarouter
   python -m venv .venv
   .venv\Scripts\activate  # Windows
   # source .venv/bin/activate  # Linux/Mac
   ```

2. **Install dependencies**:
   ```bash
   pip install -e .
   ```

3. **Start the router**:
   ```bash
   python -m metarouter.main
   ```

## Configuration

Edit `config/router.yaml`:

```yaml
server:
  host: "0.0.0.0"
  port: 8000

lm_studio:
  base_url: "http://localhost:1234"
  timeout: 300
  refresh_interval: 60  # Refresh model list every 60s

router:
  model: "microsoft/phi-4"      # Router model (keep loaded)
  prefer_loaded_bonus: 50       # Higher = stronger preference for loaded models
  auto_load_models: true        # Allow loading new models

performance_tracking:
  enabled: true                 # Learn from real inference metrics
  sample_size: 10              # Track last N inferences per model
```

## Usage

Point any OpenAI-compatible client to `http://localhost:8000` instead of `http://localhost:1234`:

### Continue.dev

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

### Python OpenAI Client

```python
from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:8000/v1",
    api_key="not-needed"
)

response = client.chat.completions.create(
    model="gpt-4",  # Ignored, router selects model
    messages=[
        {"role": "user", "content": "Explain quantum computing"}
    ]
)
print(response.choices[0].message.content)
```

### Curl

```bash
curl -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [
      {"role": "user", "content": "Write a sorting algorithm in Python"}
    ],
    "stream": false
  }'
```

## API Endpoints

- `POST /v1/chat/completions` - Main routing endpoint (OpenAI compatible)
- `GET /v1/models` - List available models
- `GET /health` - Health check
- `GET /` - API information

## Example Routing Behavior

| Query | phi-4 Decision | Reasoning |
|-------|---------------|-----------|
| "Write Python code" | qwen3-coder-30b (if loaded) | Code-specialized, already loaded |
| "Hello!" | qwen2.5-0.5b (if loaded) | Ultra-fast, sufficient for simple chat |
| "Explain quantum physics" | glm-4.7-flash or gpt-oss-120b | Knowledge/reasoning, balances quality & speed |
| "Analyze this image" | qwen3-vl-8b | Vision capability required |
| "Prove this theorem" | gpt-oss-120b | Complex reasoning, worth loading large model |

## Auto-Start on Windows Boot

### Using Docker (Easiest)

Docker Desktop will auto-start the container on Windows boot if:
1. Docker Desktop is set to start on login (Settings → General → "Start Docker Desktop when you log in")
2. Container has `restart: unless-stopped` policy (already configured)

### Using Windows Service (Without Docker)

```bash
# Install NSSM
winget install NSSM

# Create service
nssm install MetaRouter "C:\Users\Hunter\Desktop\LLM\metarouter\.venv\Scripts\python.exe"
nssm set MetaRouter AppParameters "-m metarouter.main"
nssm set MetaRouter AppDirectory "C:\Users\Hunter\Desktop\LLM\metarouter"
nssm set MetaRouter DisplayName "MetaRouter"
nssm set MetaRouter Start SERVICE_AUTO_START

# Start service
nssm start LMRouter
```

## Docker Management

```bash
# Start
docker-compose up -d

# Stop
docker-compose down

# Restart
docker-compose restart

# View logs
docker logs -f metarouter

# Rebuild after code changes
docker-compose up -d --build

# Check status
docker ps | grep metarouter
```

## Troubleshooting

### Router can't connect to LM Studio

- Ensure LM Studio is running on `http://localhost:1234`
- Check LM Studio API server is enabled (Settings → Server)
- Verify firewall isn't blocking port 1234

### phi-4 not loaded warning

- Load `microsoft/phi-4` in LM Studio UI
- Router needs phi-4 loaded to make routing decisions

### No models available

- Check LM Studio has models downloaded
- Try force refresh: restart the router

### Routing to wrong model

- Check router logs for phi-4's reasoning
- Adjust `prefer_loaded_bonus` in config for stronger/weaker loaded model preference
- Ensure models are properly tagged in LM Studio

### Performance tracking not working

- Only works for non-streaming completions
- Check `performance_tracking.enabled: true` in config
- Stats appear after first few requests to each model

## Development

### Run tests

```bash
pip install -e ".[dev]"
pytest
```

### Format code

```bash
black src
ruff check src --fix
```

## Architecture

```
metarouter/
├── src/metarouter/
│   ├── main.py                 # FastAPI app
│   ├── config/
│   │   └── settings.py         # Pydantic settings
│   ├── api/
│   │   ├── routes.py           # API endpoints
│   │   └── schemas.py          # Request/response models
│   ├── routing/
│   │   ├── router.py           # Main routing logic
│   │   └── phi4_selector.py    # Model selection logic
│   ├── lmstudio/
│   │   ├── client.py           # LM Studio API client
│   │   └── models.py           # Data models
│   └── cache/
│       └── performance.py      # Performance tracking
├── config/
│   └── router.yaml             # Configuration
├── Dockerfile
├── docker-compose.yml
├── LICENSE                     # MIT License
├── CONTRIBUTING.md
├── CHANGELOG.md
└── pyproject.toml
```

## Contributing

We welcome contributions! See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

**Areas we'd love help with:**
- Support for other router models (beyond phi-4)
- Additional backends (Ollama, vLLM, SGLang)
- Performance optimizations
- Documentation and examples
- Test coverage

## License

MIT License - see [LICENSE](LICENSE) for details.

## Acknowledgments

MetaRouter was inspired by and complements:
- [LM Studio](https://lmstudio.ai) - Local LLM inference server
- [RouteLLM](https://github.com/lm-sys/RouteLLM) - ML-based routing framework
- [Olla](https://github.com/thushan/olla) - High-performance LLM proxy
- [LiteLLM](https://github.com/BerriAI/litellm) - Multi-provider orchestration

## Support

- 📖 [Documentation](README.md)
- 🚀 [Quick Start Guide](QUICKSTART.md)
- 🐛 [Report Issues](https://github.com/yourusername/metarouter/issues)
- 💬 [Discussions](https://github.com/yourusername/metarouter/discussions)
