# Contributing to Photonic Flash Attention

Thank you for your interest in contributing to Photonic Flash Attention! This document provides guidelines and information for contributors.

## 🚀 Quick Start

1. **Fork the repository** and clone your fork
2. **Set up the development environment**:
   ```bash
   git clone https://github.com/yourusername/photonic-flash-attention.git
   cd photonic-flash-attention
   pip install -e ".[dev,benchmark]"
   ```
3. **Run the tests** to ensure everything works:
   ```bash
   pytest tests/unit/
   ```

## 🎯 Ways to Contribute

### 1. Code Contributions
- **New Features**: Photonic kernels, hardware backends, optimizations
- **Bug Fixes**: Fix issues, improve stability
- **Performance**: Optimize existing implementations
- **Documentation**: Improve code documentation and examples

### 2. Hardware Support
- **New Backends**: Add support for new photonic accelerators
- **Driver Integration**: Improve hardware detection and drivers
- **Calibration**: Develop calibration routines for photonic devices

### 3. Research & Benchmarks
- **Benchmarking**: Add new performance benchmarks
- **Validation**: Accuracy testing against reference implementations
- **Research**: Implement new photonic computing techniques

### 4. Testing & Quality
- **Test Coverage**: Add unit, integration, and performance tests
- **Security**: Identify and fix security issues
- **Documentation**: Tutorials, examples, API documentation

## 🛠️ Development Setup

### Prerequisites
- Python 3.9+
- PyTorch 2.0+
- CUDA 12.0+ (for GPU support)
- Docker (optional, for containerized development)

### Environment Setup
```bash
# Clone the repository
git clone https://github.com/danieleschmidt/photonic-mlir-synth-bridge.git
cd photonic-mlir-synth-bridge

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install in development mode
pip install -e ".[dev,benchmark,hardware]"

# Install pre-commit hooks
pre-commit install
```

### Docker Development
```bash
# Build development container
docker build -t photonic-dev --target development .

# Run development container
docker run -it --gpus all -p 8888:8888 -v $(pwd):/app photonic-dev

# Or use docker-compose
docker-compose up photonic-dev
```

## 📋 Development Guidelines

### Code Style
- **Python**: Follow PEP 8, use Black formatter (line length: 88)
- **Type Hints**: Use type hints for all public APIs
- **Docstrings**: Google-style docstrings for all public functions
- **Imports**: Use isort for import organization

### Code Quality
- **Linting**: Code must pass flake8 checks
- **Type Checking**: Use mypy for static type checking
- **Testing**: Maintain >85% test coverage
- **Security**: Run bandit security scans

### Git Workflow
1. **Create a branch** from `develop`:
   ```bash
   git checkout develop
   git pull origin develop
   git checkout -b feature/your-feature-name
   ```

2. **Make your changes** with clear, atomic commits:
   ```bash
   git add .
   git commit -m "feat: add photonic kernel optimization"
   ```

3. **Test your changes**:
   ```bash
   # Run all tests
   pytest
   
   # Run specific test categories
   pytest tests/unit/
   pytest tests/security/
   pytest tests/performance/ -m "not slow"
   ```

4. **Push and create a pull request**:
   ```bash
   git push origin feature/your-feature-name
   ```

### Commit Messages
Use [Conventional Commits](https://www.conventionalcommits.org/):
- `feat:` New features
- `fix:` Bug fixes
- `docs:` Documentation changes
- `test:` Adding tests
- `perf:` Performance improvements
- `refactor:` Code refactoring
- `chore:` Maintenance tasks

Examples:
```
feat(photonic): add LightMatter Mars backend support
fix(security): prevent tensor size overflow attack
perf(attention): optimize tiled computation for large sequences
docs(api): add photonic configuration examples
```

## 🧪 Testing

### Test Categories
- **Unit Tests**: Test individual components (`tests/unit/`)
- **Integration Tests**: Test component interactions (`tests/integration/`)
- **Performance Tests**: Benchmark performance (`tests/performance/`)
- **Security Tests**: Validate security measures (`tests/security/`)

### Running Tests
```bash
# All tests
pytest

# Unit tests only
pytest tests/unit/ -v

# With coverage
pytest --cov=src --cov-report=html

# Performance benchmarks (slow)
pytest tests/performance/ -m "performance"

# Security tests
pytest tests/security/ -m "security"

# Skip slow tests
pytest -m "not slow"
```

### Writing Tests
- **Naming**: Test functions should start with `test_`
- **Structure**: Use Arrange-Act-Assert pattern
- **Fixtures**: Use pytest fixtures for common setup
- **Parametrization**: Test multiple scenarios with `@pytest.mark.parametrize`
- **Mocking**: Mock external dependencies and hardware

Example test:
```python
import pytest
import torch
from photonic_flash_attention import PhotonicFlashAttention

class TestPhotonicAttention:
    def test_forward_pass_basic(self, device):
        """Test basic forward pass functionality."""
        # Arrange
        attention = PhotonicFlashAttention(embed_dim=512, num_heads=8)
        query = torch.randn(2, 128, 512, device=device)
        
        # Act
        output = attention(query)
        
        # Assert
        assert output.shape == query.shape
        assert torch.isfinite(output).all()
```

## 🔒 Security Guidelines

### Security Best Practices
- **Input Validation**: Validate all user inputs
- **Memory Safety**: Prevent buffer overflows and memory leaks
- **Access Control**: Use principle of least privilege
- **Secrets**: Never commit secrets or API keys
- **Dependencies**: Keep dependencies updated

### Security Testing
- Run security tests: `pytest tests/security/`
- Use bandit for static analysis: `bandit -r src/`
- Check dependencies: `safety check`
- Container scanning: `trivy image photonic-flash-attention`

## 📚 Documentation

### Documentation Types
- **API Documentation**: Docstrings in code
- **Tutorials**: Step-by-step guides
- **Examples**: Working code examples
- **Architecture**: System design documentation

### Writing Documentation
- **Clear and Concise**: Write for your audience
- **Code Examples**: Include working examples
- **Keep Updated**: Update docs with code changes
- **Test Examples**: Ensure code examples work

### Building Documentation
```bash
cd docs
make html
open _build/html/index.html
```

## 🚀 Release Process

### Version Numbering
We follow [Semantic Versioning](https://semver.org/):
- **MAJOR**: Breaking changes
- **MINOR**: New features, backward compatible
- **PATCH**: Bug fixes, backward compatible

### Release Checklist
1. Update version in `pyproject.toml`
2. Update `CHANGELOG.md`
3. Run full test suite
4. Create release branch: `release/v1.2.0`
5. Tag release: `git tag v1.2.0`
6. Push to main branch
7. GitHub Actions will handle PyPI deployment

## 🤝 Community Guidelines

### Code of Conduct
- **Be Respectful**: Treat everyone with respect
- **Be Inclusive**: Welcome all contributors
- **Be Collaborative**: Work together constructively
- **Be Professional**: Maintain professional communication

### Getting Help
- **GitHub Issues**: For bugs and feature requests
- **Discussions**: For questions and general discussion
- **Discord/Slack**: Real-time community chat (link TBD)
- **Email**: For security issues or private matters

### Review Process
1. **Automated Checks**: CI/CD pipeline runs automatically
2. **Code Review**: At least one maintainer reviews
3. **Testing**: All tests must pass
4. **Documentation**: Updates must include docs
5. **Approval**: Maintainer approval required for merge

## 🏗️ Architecture Overview

### Core Components
```
src/photonic_flash_attention/
├── core/                    # Core attention implementations
│   ├── flash_attention_3.py # GPU baseline
│   ├── photonic_attention.py # Photonic implementation
│   └── hybrid_router.py     # Intelligent routing
├── photonic/               # Photonic hardware
│   ├── optical_kernels/    # Optical computing kernels
│   ├── hardware/          # Hardware detection/drivers
│   └── simulation/        # Software simulation
├── integration/           # Framework integrations
│   └── pytorch/          # PyTorch modules
└── utils/                # Utilities
    ├── logging.py        # Logging system
    ├── validation.py     # Input validation
    └── security.py       # Security utilities
```

### Adding New Components
1. **Create Module**: Add to appropriate directory
2. **Import in `__init__.py`**: Make it discoverable
3. **Add Tests**: Comprehensive test coverage
4. **Document**: API documentation and examples
5. **Update Integration**: Add to main interfaces

## 🎯 Specific Contribution Areas

### 1. Photonic Hardware Backends
**Goal**: Support new photonic accelerators

**Tasks**:
- Implement hardware detection
- Add device drivers
- Create optical kernels
- Add calibration routines

**Files to modify**:
- `src/photonic_flash_attention/photonic/hardware/`
- `src/photonic_flash_attention/photonic/optical_kernels/`

### 2. Performance Optimization
**Goal**: Improve computation speed and efficiency

**Tasks**:
- Optimize memory usage
- Improve kernel implementations
- Add caching mechanisms
- Parallel processing improvements

**Files to modify**:
- `src/photonic_flash_attention/core/`
- `src/photonic_flash_attention/optimization/`

### 3. Framework Integration
**Goal**: Support more ML frameworks

**Tasks**:
- Add JAX/Flax support
- TensorFlow/Keras integration
- ONNX export support
- MLX integration

**Files to modify**:
- `src/photonic_flash_attention/integration/`

### 4. Research Features
**Goal**: Implement cutting-edge research

**Tasks**:
- New attention mechanisms
- Advanced photonic algorithms
- Hybrid computing strategies
- Quantum-photonic interfaces

**Files to modify**:
- `src/photonic_flash_attention/research/` (new)

## 📊 Performance Benchmarking

### Benchmark Categories
- **Latency**: Time per forward pass
- **Throughput**: Tokens processed per second
- **Memory**: Peak memory usage
- **Energy**: Power consumption
- **Accuracy**: Numerical precision

### Adding Benchmarks
```python
@pytest.mark.performance
@pytest.mark.slow
def test_new_benchmark(benchmark_results):
    """Add your benchmark here."""
    # Implement benchmark
    # Store results in benchmark_results
    pass
```

## 🔧 Hardware Testing

### Simulation Mode
For contributors without photonic hardware:
```bash
export PHOTONIC_SIMULATION=1
pytest tests/
```

### Hardware Access
- **Cloud GPUs**: Use cloud providers for GPU testing
- **Photonic Hardware**: Contact maintainers for access
- **CI/CD**: GitHub Actions provides GPU runners

## 📈 Monitoring Contributions

### Metrics We Track
- **Code Quality**: Test coverage, complexity
- **Performance**: Benchmark results over time
- **Security**: Vulnerability scans
- **Usage**: API usage patterns

### Contribution Recognition
- **Contributors.md**: All contributors listed
- **Release Notes**: Significant contributions noted
- **Badges**: Special recognition for major contributions

## 🚨 Reporting Issues

### Bug Reports
Use the bug report template:
- **Environment**: OS, Python version, hardware
- **Expected Behavior**: What should happen
- **Actual Behavior**: What actually happens
- **Reproduction Steps**: How to reproduce
- **Logs**: Relevant error messages

### Feature Requests
Use the feature request template:
- **Problem**: What problem does this solve?
- **Solution**: Proposed solution
- **Alternatives**: Alternative solutions considered
- **Use Cases**: Who would benefit?

### Security Issues
**DO NOT** create public issues for security vulnerabilities.
Email: security@terragonlabs.ai

## 🎉 Recognition

### Hall of Fame
We recognize contributors in several ways:
- **Contributors.md**: Permanent recognition
- **Release Notes**: Feature attribution
- **Conference Talks**: Present your contributions
- **Research Papers**: Co-authorship opportunities

### Contribution Levels
- **🌟 Contributor**: Made a merged contribution
- **⭐ Regular Contributor**: 5+ merged contributions
- **🏆 Core Contributor**: 20+ contributions, design input
- **👑 Maintainer**: Ongoing project stewardship

---

Thank you for contributing to Photonic Flash Attention! Together, we're building the future of photonic-accelerated machine learning. 🚀✨