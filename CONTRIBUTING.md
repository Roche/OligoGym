# Contributing to OligoGym

Thank you for your interest in contributing to OligoGym! This document provides guidelines and instructions for contributing to this project.

## Getting Started

### Prerequisites
- Git
- Python 3.10 or higher
- Any additional dependencies

### Setting Up Development Environment
1. Fork the repository
2. Clone your fork: `git clone https://github.com/Roche/oligogym.git`
3. Navigate to the project directory: `cd oligogym`
4. Install dependencies using Poetry: `poetry install --with dev`

## How to Contribute

### Reporting Bugs
- Check if the bug has already been reported in the Issues tab
- Use the bug report template
- Include detailed steps to reproduce the issue
- Provide system information and environment details

### Suggesting Features
- Check existing issues to avoid duplicates
- Use the feature request template
- Clearly describe the feature and its benefits

### Pull Requests
1. Create a branch: `git checkout -b feature/your-feature-name`
2. Make your changes
3. Format code: `poetry run black oligogym/ tests/`
4. Lint code: `poetry run flake8 oligogym/ tests/`
5. Run tests: `poetry run pytest`
6. Commit with clear messages: `git commit -m "Add feature X"`
7. Push to your fork: `git push origin feature/your-feature-name`
8. Open a pull request against the `main` branch

## Code Standards
- Follow PEP 8 style guidelines for Python code
- Write docstrings for all functions, classes, and modules
- Keep code modular and maintainable
- Add appropriate tests for new features

## Code Review Process
- All submissions require review
- Maintainers will review your PR as soon as possible
- Address any requested changes promptly

## Community
- Be respectful and inclusive
- Help others when possible

## License
By contributing, you agree that your contributions will be licensed under the project's license.

Thank you for contributing to OligoGym!