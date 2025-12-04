# Testing Architecture

## Overview

SpectralMC's test harness is GPU-only and runs through `poetry run test-all`. Output must be
redirected to files (see `documents/engineering/testing_requirements.md`) so failures are fully
captured. This document focuses on timeout behavior to prevent hung suites without masking long
GPU work.

## Per-Test Timeout Contract

- **Default**: Every test (unit, integration, e2e) is capped at **60 seconds** including setup and
  teardown. The autouse fixture in `tests/conftest.py` enforces this via `SIGALRM`.
- **Overrides**: Use `@pytest.mark.timeout(seconds=...)` only for validated long cases. Timeouts
  must remain positive and should be justified in-code to avoid casual increases.
- **Coverage**: The timeout guards async tests, storage CLI calls, and GPU kernels to prevent hung
  CUDA operations or network calls from blocking the suite.

## Suite Execution

- Avoid short suite-level shell timeouts so `poetry run test-all` can complete and logs remain
  intact; rely on the per-test guard to prevent hangs.

## References

- Autouse timeout fixture: `tests/conftest.py`
- Execution and log capture: `documents/engineering/testing_requirements.md`
- Agent guidance: `AGENTS.md`, `codex.toml`
