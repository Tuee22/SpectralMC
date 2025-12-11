# File: documents/documentation_standards.md
# Documentation Standards

**Status**: Authoritative source  
**Supersedes**: documents/engineering/documentation_standards.md  
**Referenced by**: CLAUDE.md, documents/engineering/README.md, documents/engineering/coding_standards.md, documents/engineering/pydantic_patterns.md

> **Purpose**: Single source of truth for SpectralMC documentation. Aligns with the BBY
> framework while enforcing SpectralMC’s GPU-first and type-safe constraints.

---

## Cross-References

- [engineering/README.md](engineering/README.md) — Engineering standards hub and SSoT pointers
- [documents/README.md](README.md) — Documentation navigation and folder layout
- [CLAUDE.md](../CLAUDE.md) — Agent quick reference aligned to this SSoT

## 1. Philosophy and SSoT

- **SSoT-first**: One canonical document per topic; overlays list *only* deltas and link back.
- **DRY + link**: Prefer deep relative links and bidirectional links for paired docs.
- **Separation of concerns**: Standards (how to build), domain (algorithms), tutorials/product,
  and generated references.
- **Snake_case in this repo**: BBY prefers kebab-case, but SpectralMC standardizes on
  snake_case; do not mix styles inside this repo.

## 2. Tiered Structure and Canonical Docs

1) Framework standards (this document)  
2) Core SSoTs (architecture/prereqs/code quality)  
3) Patterns/how-tos (command patterns, deployment guides)  
4) Generated reference (command-reference, metrics)

**SpectralMC canonical SSoTs**

| Document | Canonical for |
|----------|---------------|
| [coding_standards.md](engineering/coding_standards.md) | Type safety, ADTs, error handling, deprecation |
| [testing_requirements.md](engineering/testing_requirements.md) | GPU-only tests, determinism, anti-patterns |
| [testing_architecture.md](engineering/testing_architecture.md) | Test organization, DRY patterns, fixtures |
| [immutability_doctrine.md](engineering/immutability_doctrine.md) | Immutability rules |
| [pytorch_facade.md](engineering/pytorch_facade.md) | Determinism, CVNN patterns |
| [cpu_gpu_compute_policy.md](engineering/cpu_gpu_compute_policy.md) | Device placement boundaries |
| [docker_build_philosophy.md](engineering/docker_build_philosophy.md) | Build strategy and topology |
| [blockchain_storage.md](engineering/blockchain_storage.md) | Model versioning and atomic commits |
| [effect_interpreter.md](engineering/effect_interpreter.md) | Effect ADTs and interpreter patterns |
| [total_pure_modelling.md](engineering/total_pure_modelling.md) | Total ADTs and pure state machines |
| [reproducibility_proofs.md](engineering/reproducibility_proofs.md) | Determinism proofs |

Overlays must link to the canonical doc first, then list overrides only.

## 3. Naming and Layout

- Snake_case for file names within SpectralMC (e.g., `documentation_standards.md`).
- Descriptive, unabbreviated names; no version numbers in filenames.
- Keep prefixes for grouping when needed (e.g., `gpu_`, `cli_`).
- Explicit exceptions for compatibility: top-level `README.md`, `CLAUDE.md`, and `AGENTS.md`
  stay uppercase; keep references consistent.

## 4. Required Header Metadata

Every doc begins with:

```markdown
# File: documents/example_path.md
# Document Title

**Status**: [Authoritative source | Reference only | Deprecated]
**Supersedes**: prior version (if applicable)
**Referenced by**: related documents (when known)

> **Purpose**: 1-2 sentence role of the doc.
> **📖 Authoritative Reference**: [link] (if Status = Reference only)
```

Authoritative docs add a **Cross-References** section. Do not add dates; git history is
the recency signal.

## 5. Cross-Referencing and Duplication

- Use relative links with anchors; prefer deep links. Bidirectional links when documents
  depend on each other.
- Overlays: link to the base SSoT first, then list deltas only.
- Run link validation after moves/renames; avoid dead links.

Duplication rules:
- **Allowed**: navigation tables, short summaries (≤5 lines) with a link.
- **Forbidden**: copied examples/diagrams/procedures, restated policies without attribution.

## 6. Code Examples (Markdown)

- Always include a language fence; first line is a file-path comment.
- Prefer WRONG/RIGHT pairs for anti-patterns.
- Zero tolerance for `Any`, `cast`, or `# type: ignore` in examples.
- Keep lines readable (Black’s 88-character wrap is the target).

```python
# File: tools/cli/effects.py
# ❌ WRONG - Mutable domain model
@dataclass
class User:
    name: str
    email: str

# ✅ CORRECT - Frozen for immutability
@dataclass(frozen=True)
class User:
    name: str
    email: str
```

## 7. Docstrings and API Documentation

**Google-style docstrings are mandatory** for public modules, classes, and functions.

- Sections: Args, Returns, Raises (and Yields/Note/Example when relevant).
- Describe shapes/types in prose even with type hints.
- Wrap at ~88 characters; include examples (doctest or fenced code).
- Effectful functions add an **Effects** subsection (e.g., I/O, GPU, network).

Module docstrings:
- Overview + implementation details (precision/device/purity).
- Catalogue/API table listing main classes/functions.

Class docstrings:
- Overview + Attributes + example usage.
- `__init__` docstring required.

Function docstrings:
- One-line summary, blank line, Google sections, shapes/constraints, math blocks allowed.

## 8. ADT and Effect Documentation

- Document the discriminator (`kind`) and enumerate all variants with fields and types.
- Provide an exhaustive match example; link to `coding_standards.md` and
  `effect_interpreter.md`.
- Effect-producing APIs must list all side effects under **Effects** and prefer an
  effect-flow Mermaid diagram when non-trivial.

## 9. Mermaid Diagram Standards

- Orientation: **TB** by default; LR only for ≤3 sequential steps or parallel-only flows.
- Allowed: `flowchart TB|LR`, nodes `[ ] ( ) { }`, solid arrows `-->`.
- Labels: simple words, avoid punctuation/hyphens; do not mix arrow-label styles within a
  single diagram.
- Forbidden: dotted lines, subgraphs, thick arrows, `Note over`, mixed arrow styles, and
  right-to-left LR flows.
- Validation: preview in GitHub, VSCode (Mermaid extension), and Mermaid Live.

## 10. Templates

**Authoritative Doc**
```markdown
# File: documents/example.md
# Title
**Status**: Authoritative source
**Supersedes**: prior version
**Referenced by**: related documents

> **Purpose**: SSoT for [topic].

## Executive Summary
[1-2 paragraphs]

## Cross-References
- [Related Standard](engineering/coding_standards.md#result-types)
- [Upstream Architecture](engineering/effect_interpreter.md)
```

**Reference Doc**
```markdown
# File: documents/example.md
# Title
**Status**: Reference only
> **📖 Authoritative Reference**: [Canonical Doc](documentation_standards.md)

## Quick Summary
[short overview + links]
```

Tutorials and API references may reuse Effectful templates; include prerequisites, numbered
steps, examples, and next steps where applicable.

## 11. Maintenance Checklist

- [ ] Update the authoritative source first; then overlays/dependent links.
- [ ] Run link validation after renames/moves.
- [ ] Validate Mermaid diagrams across GitHub, VSCode, Mermaid Live.
- [ ] Regenerate generated docs when commands/tests change.
- [ ] Sync CLAUDE.md/indices when paths change.
- [ ] Quarterly freshness review for examples/diagrams/patterns.
- [ ] No dated changelogs; rely on git history.

## 12. Anti-Patterns

- Vague status values (“WIP”) or missing metadata blocks.
- Mixing naming styles within the repo.
- Copy-pasted diagrams or procedures; duplicated examples without attribution.
- Arrow-style mixing inside a diagram; labels with punctuation/hyphens.
- Docstrings missing Args/Returns/Raises or shapes/types.
- Examples using `Any`, `cast`, `# type: ignore`, or ellipses without context.

## 13. SpectralMC-Specific Notes

- Tests and docs assume GPU availability and deterministic execution; seed randomness.
- Whitepapers stay in `documents/whitepapers/` and cover theory or architecture decisions;
  do not duplicate docstrings or standards already covered elsewhere.
- Keep purity boundaries explicit: effectful modules declare effects and provide effect-flow
  diagrams when helpful.
