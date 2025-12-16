# File: documents/engineering/user_automation_reference.md
# User Automation Reference

**Status**: Authoritative source  
**Supersedes**: Prior automation templates  
**Referenced by**: purity_enforcement.md, CLAUDE.md

> **Purpose**: Document the prohibition on repository-managed automation; all validation and checks must be run manually by contributors.
> **📖 Authoritative Reference**: [documentation_standards.md](documentation_standards.md#13-automation-prohibitions-authoritative)

---

## Cross-References

- [documentation_standards.md](documentation_standards.md#13-automation-prohibitions-authoritative) — SSoT for documentation policy
- [purity_enforcement.md](purity_enforcement.md) — Manual purity enforcement CLI
- [CLAUDE.md](../../CLAUDE.md) — Agent workflow guardrails

---

## Doctrine: No Repository Automation

- Do not add `.pre-commit-config.yaml`, git hooks, or any git-managed automation to this repository.
- Do not add `.github/workflows/` or other GitHub Actions/GitLab CI/CD pipelines.
- Remove any automation artifacts discovered in the working tree; validation is manual-only.

---

## Manual Verification Contract

Run checks explicitly (inside Docker) before sharing changes:

```bash
# File: tools/manual_verification.sh
# Format
docker compose -f docker/docker-compose.yml exec spectralmc black .

# Lint + type + purity + sync checks
docker compose -f docker/docker-compose.yml exec spectralmc poetry run check-code

# Tests (GPU required)
docker compose -f docker/docker-compose.yml exec spectralmc poetry run test-all
```

- No automated hooks or pipelines may wrap these commands.
- Capture outputs manually when preparing handoffs or reviews.

---

## Responsibilities

- **Contributors**: Run the manual commands above; do not introduce automation files or CI/CD hooks.
- **Claude Code**: Must not author automation configs or suggest adding them; must flag and remove them when encountered.
