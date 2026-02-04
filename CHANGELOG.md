# Changelog

All notable changes to MetaRouter will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.1.0] - 2025-02-03

### Added
- Initial release of MetaRouter
- LLM-based routing using phi-4 for intelligent model selection
- Dynamic model discovery from LM Studio API
- Zero-configuration setup - automatically detects available models
- Performance tracking and learning from real inference metrics
- OpenAI-compatible API (`/v1/chat/completions`, `/v1/models`)
- Full streaming support with SSE passthrough
- Docker deployment with auto-restart
- Latency-aware routing - strongly prefers already-loaded models
- Comprehensive documentation (README, QUICKSTART, CONTRIBUTING)
- MIT License
- Python 3.12+ support with latest dependencies
- FastAPI-based proxy server
- Health check endpoint
- CORS middleware for cross-origin requests

### Features
- **Intelligent Routing**: Uses phi-4 to understand query semantics and select optimal models
- **Self-Learning**: Tracks tokens/sec, time-to-first-token, and generation time
- **Model Context**: Provides router with model capabilities, load states, and performance data
- **Explainable**: Router provides reasoning for each model selection
- **Automatic Discovery**: Finds new models as you install them in LM Studio
- **Configuration**: Simple YAML config for server, LM Studio connection, and router settings

[unreleased]: https://github.com/yourusername/metarouter/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/yourusername/metarouter/releases/tag/v0.1.0
