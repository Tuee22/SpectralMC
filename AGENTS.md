# File: AGENTS.md
# Repository Guidelines

**Status**: Authoritative source  
**Supersedes**: None  
**Referenced by**: CLAUDE.md; README.md

> **Purpose**: Contribution guardrails for SpectralMC agents and developers (structure, commands, safety).

## Project Structure & Module Organization
- `src/spectralmc/`: Core runtime (effects, models, storage, serialization, proto bindings).
- `tests/`: GPU-only unit/integration suites (`tests/test_effects`, `tests/test_storage`, `tests/test_e2e`, `tests/test_integrity`).
- `examples/`: Minimal runnable samples; mirror patterns used in `src/`.
- `documents/`: Engineering/product standards (coding, purity, testing, GPU policies).
- `docker/`: Container builds, CUDA validation, and developer bootstrap.
- `stubs/`: Strict third-party `.pyi` stubs (no `Any`, no `cast`, no `type: ignore`).
- `notebooks/`: Exploratory experiments; keep outputs clean.

## Build, Test, and Development Commands
- Start dev container: `cd docker && docker compose up -d`.
- Full lint/format/type sweep: `docker compose -f docker/docker-compose.yml exec spectralmc poetry run check-code`.
- Type-only pass: `docker compose -f docker/docker-compose.yml exec spectralmc mypy`.
- Test runner (GPU required): `docker compose -f docker/docker-compose.yml exec spectralmc poetry run test-all > /tmp/test-output.txt 2>&1`.
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
- Default per-test timeout is 60s via autouse fixture; use `@pytest.mark.timeout(seconds=...)` only when justified.
- Codex/automation must wrap test commands with a timeout no lower than 4 hours to avoid hanging agents.
- Organize new cases alongside the feature’s area; mirror examples from `tests/test_storage` or `tests/test_effects`.
- Capture and attach test logs from `poetry run test-all` in PRs for reviewability.

## 🔒 Git Workflow Policy for LLMs

**CRITICAL**: LLMs (including Claude Code, GitHub Copilot, and all AI assistants) are STRICTLY FORBIDDEN from making commits, creating branches, or pushing changes.

### Absolutely Forbidden Operations
- ❌ **NEVER** run `git commit` (any variant: `--amend`, `--no-verify`, etc.)
- ❌ **NEVER** run `git push` (any variant: `--force`, `--force-with-lease`, etc.)
- ❌ **NEVER** run `git checkout -b` or `git branch <name>` to create branches
- ❌ **NEVER** run `git switch -c` to create branches
- ❌ **NEVER** run `git add` followed by commit operations
- ❌ **NEVER** modify git history (rebase, reset, amend)

### Required LLM Workflow
- ✅ Make all requested code changes
- ✅ Run validation: `check-code` and `test-all`
- ✅ Leave ALL changes as uncommitted working directory changes
- ✅ Human reviews with `git status` and `git diff`
- ✅ Human manually creates branches, commits, and pushes

### Human Commit & Pull Request Guidelines
When the human creates commits and PRs (NOT the LLM):
- Use small, focused commits with imperative summaries (e.g., "Add Sobol sampler fixture")
- Include what/why in PR descriptions, linked issues if applicable, and note GPU environment used
- Show validation: paste the `check-code` and `test-all` command outputs or attach the log file path
- Update docs/examples when behavior or APIs change; ensure stubs stay in sync with new library usage
- Avoid adding secrets; keep configuration in environment variables or `.env` files not tracked in git

## Security & Configuration Tips
- S3/chain credentials belong in runtime env/config, never in source control.
- When adding dependencies that need stubs, place them in `stubs/` and keep them type-pure.
- Prefer immutability and expression-style control flow per documents in `documents/engineering/`.
