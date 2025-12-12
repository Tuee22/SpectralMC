# File: documents/engineering/user_automation_reference.md
# User Automation Reference

**Status**: Reference only
**Supersedes**: None
**Referenced by**: purity_enforcement.md, CLAUDE.md

> **Purpose**: Optional automation templates for users who want to enforce purity checks via git hooks or GitHub Actions. These are user-configured and NOT automated by Claude Code.
> **📖 Authoritative Reference**: [purity_enforcement.md](purity_enforcement.md) for enforcement rules

---

## ⚠️ IMPORTANT DISCLAIMER

**User Responsibility**: All automation in this document is for users to implement and maintain manually. Per CLAUDE.md Git Workflow Policy:

- ❌ Claude Code does NOT configure git hooks
- ❌ Claude Code does NOT create GitHub Actions workflows
- ❌ Claude Code does NOT run `git commit` or `git push`
- ✅ Users configure git/GitHub automation independently
- ✅ Claude Code only implements the `check_purity.py` tool

---

## Cross-References

- [purity_enforcement.md](purity_enforcement.md) — Purity rules and AST detection
- [CLAUDE.md](../../CLAUDE.md) — Git Workflow Policy (no automated commits)
- [documentation_standards.md](documentation_standards.md) — SSoT principles

---

## Section 1: Pre-Commit Hooks (Optional)

**Purpose**: Run purity checks before allowing git commits

**Setup** (user must perform manually):

### 1.1 Installation

```bash
# File: documents/engineering/user_automation_reference.md
# Inside Docker container
docker compose -f docker/docker-compose.yml exec spectralmc poetry add --group dev pre-commit
docker compose -f docker/docker-compose.yml exec spectralmc poetry run pre-commit install
```

### 1.2 Configuration

**File**: `.pre-commit-config.yaml` (create in project root)

```yaml
# File: .pre-commit-config.yaml
repos:
  - repo: https://github.com/astral-sh/ruff-pre-commit
    rev: v0.14.8
    hooks:
      - id: ruff
        args: [--fix, --exit-non-zero-on-fix]
        files: ^(src/spectralmc|tests|tools)/.*\.py$

  - repo: https://github.com/psf/black
    rev: 25.1.0
    hooks:
      - id: black
        files: ^(src/spectralmc|tests|tools)/.*\.py$

  - repo: local
    hooks:
      - id: purity-check
        name: Purity checker (Tier 2 business logic)
        entry: poetry run check-purity
        language: system
        files: ^src/spectralmc/.*\.py$
        pass_filenames: false

  - repo: https://github.com/pre-commit/mirrors-mypy
    rev: v1.15.0
    hooks:
      - id: mypy
        files: ^(src/spectralmc|tests|tools)/.*\.py$
```

**Note**: All Poetry scripts (check-code, test-all, check-purity) are defined in `[tool.poetry.scripts]`, which is a shared section in both pyproject.binary.toml and pyproject.source.toml. Pre-commit hooks work identically for both build modes.

### 1.3 Usage

Hooks run automatically on `git commit`. To bypass (NOT RECOMMENDED):
```bash
# File: documents/engineering/user_automation_reference.md
git commit --no-verify
```

To run manually on all files:
```bash
# File: documents/engineering/user_automation_reference.md
docker compose -f docker/docker-compose.yml exec spectralmc poetry run pre-commit run --all-files
```

---

## Section 2: GitHub Actions (Optional)

**Purpose**: Run purity checks on pull requests

**Setup** (repository owner must perform manually):

### 2.1 Create Workflow File

**File**: `.github/workflows/purity-check.yml`

```yaml
# File: .github/workflows/purity-check.yml
name: Purity Compliance Check

on:
  pull_request:
    branches: [main, develop]
    paths:
      - 'src/spectralmc/**/*.py'
      - 'tools/purity/**/*.py'
      - 'pyproject.toml'

jobs:
  purity-check:
    name: Check Business Logic Purity
    runs-on: ubuntu-latest

    steps:
      - name: Checkout code
        uses: actions/checkout@v4

      - name: Set up Python
        uses: actions/setup-python@v5
        with:
          python-version: '3.12'

      - name: Install Poetry
        uses: snok/install-poetry@v1
        with:
          version: 1.8.0

      - name: Install dependencies
        run: poetry install --no-interaction

      - name: Run purity checker
        id: purity
        run: |
          poetry run check-purity --verbose > purity-report.txt 2>&1
          echo "exit_code=$?" >> $GITHUB_OUTPUT
        continue-on-error: true

      - name: Comment on PR if violations
        if: failure() && github.event_name == 'pull_request'
        uses: actions/github-script@v7
        with:
          script: |
            const fs = require('fs');
            const report = fs.readFileSync('purity-report.txt', 'utf8');

            github.rest.issues.createComment({
              issue_number: context.issue.number,
              owner: context.repo.owner,
              repo: context.repo.repo,
              body: `## ❌ Purity Violations Found\n\n\`\`\`\n${report}\n\`\`\``
            });

      - name: Fail if violations found
        if: steps.purity.outputs.exit_code != '0'
        run: exit 1
```

### 2.2 Branch Protection (Optional)

**Setup** (via GitHub repository settings):

1. Navigate to: Settings → Branches → Branch protection rules
2. Select branch: `main` or `develop`
3. Enable: "Require status checks to pass before merging"
4. Add required check: `purity-check / Check Business Logic Purity`

---

## Section 3: Troubleshooting

### 3.1 Pre-Commit Hook Failures

**Problem**: Hooks fail inside Docker container

**Solution**:
```bash
# File: documents/engineering/user_automation_reference.md
# Ensure Poetry environment is activated
docker compose -f docker/docker-compose.yml exec spectralmc poetry run pre-commit run --all-files

# Debug specific hook
docker compose -f docker/docker-compose.yml exec spectralmc poetry run check-purity --verbose
```

**Alternative**: If hooks need to run inside Docker:
```yaml
# File: .pre-commit-config.yaml
- repo: local
  hooks:
    - id: purity-check
      entry: docker compose -f docker/docker-compose.yml exec -T spectralmc poetry run check-purity
      language: system
```

### 3.2 GitHub Actions Failures

**Problem**: Workflow fails to install dependencies

**Solution**: Verify `pyproject.toml` and `poetry.lock` are committed and up-to-date.

**Problem**: Workflow runs on unrelated file changes

**Solution**: Review the `paths:` filter in workflow `on:` section to ensure it only triggers on relevant files.

---

## Section 4: Maintenance

**User Responsibilities**:
- Update `.pre-commit-config.yaml` when purity checker changes
- Monitor GitHub Actions workflow for failures
- Configure branch protection rules per repository policy
- Keep automation in sync with `check_purity.py` CLI changes
- Review and update hook versions periodically

**Claude Code does NOT**:
- Modify `.pre-commit-config.yaml`
- Create or update GitHub Actions workflows
- Configure branch protection rules
- Commit or push automation configuration

---

## Version History

**v1.0** (2025-12-10): Initial version extracted from purity_enforcement.md following SSoT refactor

---