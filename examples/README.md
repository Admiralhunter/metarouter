# MetaRouter Examples

This directory contains examples of using MetaRouter with various clients and tools.

## Files

- **`python_client.py`** - Using MetaRouter with Python OpenAI client
- **`streaming_example.py`** - Streaming responses example
- **`continue_dev.md`** - Integrating with Continue.dev code assistant
- **`curl_examples.sh`** - Command-line examples using curl

## Prerequisites

Before running these examples:

1. **LM Studio running** on `http://localhost:1234`
2. **phi-4 loaded** in LM Studio (for routing)
3. **MetaRouter running** on `http://localhost:8000`

```bash
# Start MetaRouter
cd metarouter
docker-compose up -d

# Or without Docker
python -m metarouter.main
```

## Running Examples

### Python Client Example

```bash
# Install OpenAI Python client
pip install openai

# Run examples
python examples/python_client.py
```

This demonstrates:
- Simple chat
- Code generation
- Complex reasoning
- Streaming
- Multi-turn conversations

### Streaming Example

```bash
# Install dependencies
pip install httpx

# Run streaming example
python examples/streaming_example.py
```

Shows how to handle Server-Sent Events (SSE) streaming responses.

### curl Examples

```bash
# Make script executable
chmod +x examples/curl_examples.sh

# Run all examples
./examples/curl_examples.sh
```

Demonstrates REST API usage with curl including:
- Health checks
- Model listing
- Chat completions
- Streaming
- Parameters

### Continue.dev Integration

See `continue_dev.md` for detailed instructions on integrating MetaRouter with the Continue.dev VS Code extension.

## What You'll See

When you run these examples, you'll notice:

1. **Different models for different queries**
   - Simple queries → fast small models
   - Code queries → specialized coding models
   - Reasoning queries → larger models

2. **Model selection reasoning**
   Check MetaRouter logs to see phi-4's reasoning:
   ```bash
   docker logs -f metarouter
   ```

3. **Performance metrics**
   MetaRouter tracks real performance and uses it for future routing decisions

## Creating Your Own

To use MetaRouter in your application:

```python
from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:8000/v1",
    api_key="not-needed"
)

# That's it! MetaRouter handles model selection
response = client.chat.completions.create(
    model="gpt-4",  # Ignored
    messages=[{"role": "user", "content": "Your query"}]
)
```

Or with any OpenAI-compatible client - just point it to `http://localhost:8000/v1`.

## Troubleshooting

**Examples fail to connect:**
- Verify MetaRouter is running: `curl http://localhost:8000/health`
- Check LM Studio is running: `curl http://localhost:1234/v1/models`

**Slow responses:**
- Check which models are loaded in LM Studio
- MetaRouter prefers loaded models
- View logs to see which model was selected

**Unexpected model selection:**
- Check router logs for phi-4's reasoning
- Adjust `prefer_loaded_bonus` in `config/router.yaml`

## More Resources

- [Main README](../README.md) - Full documentation
- [Quick Start Guide](../QUICKSTART.md) - Setup instructions
- [Contributing](../CONTRIBUTING.md) - How to contribute
