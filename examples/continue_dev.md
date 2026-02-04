# Using MetaRouter with Continue.dev

Continue.dev is an open-source AI code assistant. Here's how to configure it to use MetaRouter.

## Configuration

1. **Open Continue configuration**
   - VS Code: `Cmd/Ctrl+Shift+P` → "Continue: Open config.json"
   - Or edit `~/.continue/config.json` directly

2. **Add MetaRouter as a model provider**

```json
{
  "models": [
    {
      "title": "MetaRouter",
      "provider": "openai",
      "model": "gpt-4",
      "apiBase": "http://localhost:8000/v1",
      "apiKey": "not-needed"
    }
  ],
  "tabAutocompleteModel": {
    "title": "MetaRouter Fast",
    "provider": "openai",
    "model": "gpt-3.5-turbo",
    "apiBase": "http://localhost:8000/v1",
    "apiKey": "not-needed"
  }
}
```

## What This Does

- **Main chat**: Routes to optimal model based on your question
  - Code questions → coding models
  - Explanations → reasoning models
  - Simple queries → fast small models

- **Tab autocomplete**: Uses fast model for low-latency completions

## Benefits

✅ **Intelligent routing**: Different queries use different models automatically
✅ **No model switching**: MetaRouter selects the best model for each query
✅ **Cost-free**: All models are local
✅ **Fast**: Prefers already-loaded models to minimize latency

## Example Workflow

1. Ask "Explain this code" → Routes to reasoning model
2. Ask "Add error handling" → Routes to coding model
3. Ask "What does this do?" → Routes to fast model (simple query)
4. Tab complete → Always uses fast model

All automatic, zero configuration required!

## Troubleshooting

### Continue can't connect to MetaRouter

- Ensure MetaRouter is running: `docker ps | grep metarouter`
- Check logs: `docker logs -f metarouter`
- Verify port 8000 is accessible

### Completions are slow

- Check which models are loaded in LM Studio
- MetaRouter prefers loaded models - keep common models loaded
- Check router logs to see which model is being selected

### Want to force a specific model?

You can add multiple entries in Continue config:

```json
{
  "models": [
    {
      "title": "MetaRouter (Auto)",
      "provider": "openai",
      "model": "gpt-4",
      "apiBase": "http://localhost:8000/v1"
    },
    {
      "title": "Direct: Coding Model",
      "provider": "openai",
      "model": "qwen3-coder-30b",
      "apiBase": "http://localhost:1234/v1"
    }
  ]
}
```

This gives you both automatic routing and direct model access when needed.
