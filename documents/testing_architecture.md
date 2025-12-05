# File: documents/testing_architecture.md
# Testing Architecture

**Status**: Reference only  
**Supersedes**: Prior testing architecture notes  
**Referenced by**: documents/README.md; documents/engineering/testing_requirements.md

> **Purpose**: Redirect to the authoritative testing requirements SSoT.  
> **📖 Authoritative Reference**: [engineering/testing_requirements.md](engineering/testing_requirements.md)

## Quick Summary

- Per-test 60s autouse timeout in `tests/conftest.py` covering setup/teardown, async tests,
  storage CLI calls, and GPU kernels.
- Use `@pytest.mark.timeout(seconds=...)` only with inline justification and minimal positive
  values.
- Do not wrap `poetry run test-all` with shell timeouts; rely on the per-test guard and captured
  logs.
