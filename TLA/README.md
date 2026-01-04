# File: TLA/README.md
# SpectralMC TLA+ Specs

**Status**: Reference only  
**Supersedes**: None  
**Referenced by**: documents/engineering/tla.md

> **Purpose**: Root directory for SpectralMC TLA+ specifications and configs.
> **📖 Authoritative Reference**: [documents/engineering/tla.md](../documents/engineering/tla.md)

## Layout

```text
# File: TLA/README.md
TLA/
├── README.md
├── common/
│   ├── types.tla
│   ├── hashing.tla
│   └── effects.tla
├── storage/
│   ├── object_store_spec.tla
│   └── object_store_spec.cfg
├── interpreter/
│   ├── effect_interpreter.tla
│   └── effect_interpreter.cfg
├── training/
│   ├── reproducibility.tla
│   └── reproducibility.cfg
└── integration/
    ├── training_with_storage.tla
    └── training_with_storage.cfg
```

## Running TLC (Manual)

All commands must run in Docker per repo policy. See
[documents/engineering/tla.md](../documents/engineering/tla.md) for the canonical
workflow and command examples.
