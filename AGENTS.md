# Repository Guidelines

This guide keeps contributions to SpectralMC consistent, type-safe, and GPU-ready.

## Project Structure & Module Organization
- `src/spectralmc/`: Core runtime (effects, models, storage, serialization, proto bindings).
- `tests/`: GPU-only unit/integration suites (`tests/test_effects`, `tests/test_storage`, `tests/test_e2e`, `tests/test_integrity`).
- `examples/`: Minimal runnable samples; mirror patterns used in `src/`.
- `documents/`: Engineering/product standards (coding, purity, testing, GPU policies).
- `docker/` and `scripts/`: Container builds, CUDA validation, and developer bootstrap.
- `stubs/`: Strict third-party `.pyi` stubs (no `Any`, no `cast`, no `type: ignore`).
- `notebooks/`: Exploratory experiments; keep outputs clean.

## Build, Test, and Development Commands
- Start dev container: `cd docker && docker compose up -d`.
- Full lint/format/type sweep: `docker compose -f docker/docker-compose.yml exec spectralmc poetry run check-code`.
- Type-only pass: `docker compose -f docker/docker-compose.yml exec spectralmc mypy`.
- Test runner (GPU required): `docker compose -f docker/docker-compose.yml exec spectralmc poetry run test-all > /tmp/test-output.txt 2>&1`.
- Bootstrap (installs drivers, builds container, runs tests): `./scripts/build_and_run_unit_tests.sh` from repo root.
- Avoid `pytest` directly; always use `poetry run test-all` (arguments allowed).

## Coding Style & Naming Conventions
- Black with 100-char lines; Python 3.12; double quotes by default.
- No conditional/function-level imports; no `TYPE_CHECKING` guards—imports must fail fast.
- Strict typing: forbid `Any`, `cast`, and `# type: ignore`; annotate all functions/attributes.
- Modules and files use snake_case; classes are CapWords; tests named `test_*.py` with typed fixtures.

## Testing Guidelines
- All tests assume CUDA; add module-level assertions (e.g., `assert torch.cuda.is_available()`).
- Keep tests deterministic (seed PyTorch/NumPy/Numba before generating randomness).
- Prefer explicit GPU devices (`torch.device("cuda:0")`) and forbid CPU fallbacks.
- Organize new cases alongside the feature’s area; mirror examples from `tests/test_storage` or `tests/test_effects`.
- Capture and attach test logs from `poetry run test-all` in PRs for reviewability.

## Commit & Pull Request Guidelines
- Use small, focused commits with imperative summaries (e.g., “Add Sobol sampler fixture”).
- Include what/why in PR descriptions, linked issues if applicable, and note GPU environment used.
- Show validation: paste the `check-code` and `test-all` command outputs or attach the log file path.
- Update docs/examples when behavior or APIs change; ensure stubs stay in sync with new library usage.
- Avoid adding secrets; keep configuration in environment variables or `.env` files not tracked in git.

## Security & Configuration Tips
- S3/chain credentials belong in runtime env/config, never in source control.
- When adding dependencies that need stubs, place them in `stubs/` and keep them type-pure.
- Prefer immutability and expression-style control flow per documents in `documents/engineering/`.
