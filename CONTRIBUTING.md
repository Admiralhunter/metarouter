# Contributing to MetaRouter

We welcome contributions! Thank you for your interest in making MetaRouter better.

## Development Setup

1. **Fork the repository**
   ```bash
   # Fork on GitHub, then clone your fork
   git clone https://github.com/yourusername/metarouter.git
   cd metarouter
   ```

2. **Set up development environment**
   ```bash
   # Create virtual environment
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate

   # Install with dev dependencies
   pip install -e ".[dev]"
   ```

3. **Install pre-commit hooks** (optional but recommended)
   ```bash
   pip install pre-commit
   pre-commit install
   ```

## Making Changes

1. **Create a branch**
   ```bash
   git checkout -b feature/your-feature-name
   ```

2. **Make your changes**
   - Write clean, readable code
   - Add docstrings to public APIs
   - Include type hints where appropriate

3. **Run tests**
   ```bash
   pytest
   ```

4. **Format your code**
   ```bash
   black src tests
   ruff check src tests --fix
   ```

5. **Commit your changes**
   ```bash
   git add .
   git commit -m "Add feature: description of your changes"
   ```

6. **Push and create a pull request**
   ```bash
   git push origin feature/your-feature-name
   ```
   Then open a pull request on GitHub.

## Code Style

- Follow [PEP 8](https://pep8.org/)
- Use [Black](https://github.com/psf/black) for code formatting (line length: 100)
- Use [Ruff](https://github.com/astral-sh/ruff) for linting
- Add type hints to function signatures
- Write clear docstrings for public functions and classes

Example:
```python
def select_model(self, query: str) -> ModelSelection:
    """
    Select the best model for a query using the router LLM.

    Args:
        query: User's query text

    Returns:
        ModelSelection with chosen model and reasoning
    """
    ...
```

## Areas for Contribution

We're especially interested in contributions in these areas:

### 1. Router Model Support
- Support for router models beyond phi-4
- Make router model configurable
- Test with different router models (Qwen, Llama, etc.)

### 2. Additional Backends
- Ollama support
- vLLM support
- SGLang support
- Text Generation WebUI support

### 3. Performance Optimizations
- Caching improvements
- Faster model selection
- Reduced latency overhead

### 4. Features
- Model cost tracking
- Request analytics and logging
- Web dashboard for monitoring
- A/B testing between models

### 5. Documentation
- More usage examples
- Integration guides (Continue.dev, Cursor, etc.)
- Video tutorials
- Blog posts explaining the approach

### 6. Testing
- Unit test coverage
- Integration tests
- Performance benchmarks
- Edge case handling

## Pull Request Guidelines

- **One feature per PR** - Keep pull requests focused
- **Update documentation** - If you change functionality, update relevant docs
- **Add tests** - New features should include tests
- **Pass CI** - Ensure all checks pass before requesting review
- **Describe your changes** - Explain what and why in the PR description

## Bug Reports

Found a bug? Please open an issue with:
- Clear description of the problem
- Steps to reproduce
- Expected vs. actual behavior
- Environment details (OS, Python version, LM Studio version)
- Logs if applicable

## Feature Requests

Have an idea? Open an issue with:
- Description of the feature
- Use case / why it's useful
- Proposed implementation (optional)

## Questions?

- Open a discussion on GitHub
- Check existing issues and discussions
- Read the documentation in `README.md` and `QUICKSTART.md`

## Code of Conduct

- Be respectful and inclusive
- Welcome newcomers
- Focus on constructive feedback
- Help others learn and grow

Thank you for contributing to MetaRouter! 🎉
