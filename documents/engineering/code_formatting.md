# Code Formatting with Black

## Overview

All Python code in SpectralMC must be formatted with **Black 25.1+**. Black is an opinionated code formatter that enforces a consistent style across the entire codebase, eliminating debates about formatting and ensuring readability.

**Related Standards**: [Type Safety](type_safety.md), [Documentation Standards](documentation_standards.md)

---

## Running Black

Format all Python code from the project root:

```bash
# Inside Docker container
docker compose -f docker/docker-compose.yml exec spectralmc black .
```

This command will automatically format all Python files in the project, including source code, tests, and scripts.

### Checking Without Modifying

To check if files would be reformatted without actually changing them:

```bash
docker compose -f docker/docker-compose.yml exec spectralmc black --check .
```

This is useful for CI/CD pipelines to verify that all code is properly formatted before merging.

---

## Configuration

The Black version and settings are defined in `pyproject.toml`:

```toml
[tool.poetry.group.dev.dependencies]
black = ">=25.1,<26.0"
```

Black uses its default configuration with **no customization**. This is intentional:

- **Line length**: 88 characters (Black's default)
- **String quotes**: Double quotes (Black's default)
- **Trailing commas**: Added where appropriate (Black's default)
- **Import sorting**: Not handled by Black (use isort or similar if needed)

---

## Key Principles

### 1. Zero Configuration

Black's defaults are **non-negotiable**. The project does not override Black's built-in settings because:

- **Consistency**: Everyone uses the same formatting, no exceptions
- **Simplicity**: No time wasted debating formatting rules
- **Compatibility**: Works seamlessly across different environments
- **Future-proof**: Updates to Black automatically apply consistent improvements

### 2. Consistent Formatting

All code must be Black-formatted before committing. This ensures:

- **Uniform style**: All code looks like it was written by the same person
- **Minimal diffs**: Changes focus on logic, not whitespace
- **No bikeshedding**: Formatting is automated, not debated

### 3. Automated Enforcement

#### Pre-commit Hooks (Recommended)

Install pre-commit hooks to automatically format code before each commit:

```bash
# Install pre-commit (one-time setup)
docker compose -f docker/docker-compose.yml exec spectralmc pip install pre-commit

# Install git hooks (one-time setup)
docker compose -f docker/docker-compose.yml exec spectralmc pre-commit install
```

Create `.pre-commit-config.yaml` in the repository root:

```yaml
repos:
  - repo: https://github.com/psf/black
    rev: 25.1.0
    hooks:
      - id: black
        language_version: python3.12
```

Now Black will automatically format files when you run `git commit`.

#### CI/CD Integration

In continuous integration, verify formatting with:

```bash
docker compose -f docker/docker-compose.yml exec spectralmc black --check .
```

This command exits with a non-zero code if any files need formatting, failing the build.

---

## Example: Before and After

### Before Black

```python
def complex_multiply(real_a,imag_a,real_b,imag_b):
    real_result=real_a*real_b-imag_a*imag_b
    imag_result=real_a*imag_b+imag_a*real_b
    return real_result,imag_result
```

### After Black

```python
def complex_multiply(real_a, imag_a, real_b, imag_b):
    real_result = real_a * real_b - imag_a * imag_b
    imag_result = real_a * imag_b + imag_a * real_b
    return real_result, imag_result
```

Black adds:
- Spaces around operators
- Spaces after commas
- Consistent spacing

---

## Integration with mypy

Black formatting is **complementary** to mypy type checking. They serve different purposes:

- **Black**: Enforces **stylistic** consistency (whitespace, line breaks, quotes)
- **mypy**: Enforces **type** safety (annotations, type correctness)

Both must pass before code can be merged:

```bash
# Run both checks
docker compose -f docker/docker-compose.yml exec spectralmc black --check .
docker compose -f docker/docker-compose.yml exec spectralmc mypy
```

See [Type Safety](type_safety.md) for mypy configuration and requirements.

---

## Common Questions

### Q: Can I disable Black for specific lines?

**A: No.** Black does not support disabling formatting for specific code sections. This is intentional to maintain absolute consistency.

If Black's formatting seems problematic for a specific case, it's usually a sign that the code structure should be refactored for clarity.

### Q: What about import sorting?

**A: Black does not sort imports.** For import sorting, consider using `isort` with Black-compatible settings:

```toml
[tool.isort]
profile = "black"
```

However, import sorting is **not currently enforced** in SpectralMC. Black handles only whitespace and line breaks.

### Q: Can I use a different line length?

**A: No.** SpectralMC uses Black's default 88-character line length. This is Black's recommended default and works well for most code.

---

## Summary

- **Always run Black** before committing code
- **No customization** - use Black's defaults
- **Pre-commit hooks** recommended for automatic formatting
- **CI/CD checks** enforce formatting before merge
- **Complements mypy** for complete code quality

See also: [Type Safety](type_safety.md), [Testing Requirements](testing_requirements.md)
