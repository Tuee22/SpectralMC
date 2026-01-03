# File: documents/engineering/tla.md
# TLA+ Reproducibility Proofs

**Status**: Authoritative source
**Supersedes**: None
**Referenced by**: documents/documentation_standards.md; documents/engineering/README.md

> **Purpose**: Define how SpectralMC uses TLA+ to formally prove reproducibility within
> an explicit, reasonable set of assumptions, and provide a guided workflow for
> running those proofs.

## Executive Summary

This document defines the TLA+ model, assumptions, and workflow for proving that
SpectralMC training is reproducible and snapshot/restore is equivalent to
continuous training. The proof is valid only under explicit runtime and GPU
assumptions that are already enforced by existing doctrines. This document is
hands-on and includes a step-by-step workflow for running the model checker.

## Cross-References

- [Reproducibility Proofs](reproducibility_proofs.md)
- [Purity Doctrine](purity_doctrine.md)
- [Effect Interpreter](effect_interpreter.md)
- [CPU/GPU Compute Policy](cpu_gpu_compute_policy.md)
- [Torch Runtime](pytorch_facade.md)
- [Total Pure Modelling](total_pure_modelling.md)
- [Immutability Doctrine](immutability_doctrine.md)
- [Documentation Standards](../documentation_standards.md)

## Scope

The TLA+ spec covers:

- Training state transitions for `GbmCVNNPricer`
- Explicit RNG threading (torch CPU/CUDA, numpy, Sobol skip)
- Snapshot and restore equivalence
- Effect sequencing for GPU work, RNG capture/restore, and metadata updates

The spec does not model GPU kernel implementations, cuDNN internals, or external
storage engines. Those details are abstracted into assumptions.

## Explicit Assumptions

The proof holds only if all of the following are true:

1. **Deterministic runtime configuration**: deterministic flags are applied
   before any GPU work (see [pytorch_facade.md](pytorch_facade.md)).
2. **Deterministic GPU kernels**: all GPU kernels used in training are
   deterministic under the configured flags (no nondeterministic reductions or
   atomic race outcomes).
3. **Stable device topology**: CUDA device count and ordering remain stable
   across runs and restores.
4. **RNG state serialization fidelity**: serialized RNG states fully capture
   torch CPU/CUDA and numpy state, and restore reproduces the same stream.
5. **No hidden effects**: all randomness and side effects are represented as
   explicit effect ADTs (see [effect_interpreter.md](effect_interpreter.md)).
6. **Pure business logic**: Tier 2 modules remain pure per
   [purity_doctrine.md](purity_doctrine.md).

These assumptions are aligned with existing SpectralMC doctrines; the TLA+ proof
is a formalization of those contracts, not a replacement for them.

## Dependencies (Required)

TLA+ tooling is Java-based. SpectralMC provides a container-integrated setup so
proofs can run inside the standard dev container.

**Included in SpectralMC containers**:

- OpenJDK 17 runtime
- TLA+ tools jar at `/opt/tla/tla2tools.jar`
- Environment variable `TLA_JAR=/opt/tla/tla2tools.jar`

No Poetry or Python dependencies are required.
The `poetry run tla-check` entrypoint uses `TLA_JAR`; override it if you store
the jar elsewhere.

### Optional Sidecar Container (Alternative)

If you prefer not to install TLA+ in the main container, use the official TLA+
image as a sidecar. This is optional and manual; no automation is allowed.

```bash
# File: documents/engineering/tla.md
# Run TLC using a sidecar container (example)
# (Mount the repo and run TLC from /spectralmc)

docker run --rm -v "$(pwd)":/spectralmc -w /spectralmc tlaplus/tlaplus:latest \
  tlc2.TLC -workers auto -config documents/engineering/tla/reproducibility.cfg \
  documents/engineering/tla/reproducibility.tla
```

## State Model

TLA+ models the training system as a total state machine. State components map
directly to the reproducibility boundary in code.

```tla
\* File: documents/engineering/tla.md
VARIABLES
  ModelParams, OptimizerState, TrainingConfig,
  RngTorchCpu, RngTorchCuda, RngNumpy, SobolSkip,
  GlobalStep, DeviceCount, DeterminismFlags,
  EffectQueue, SnapshotStore
```

**State mapping**:

- `ModelParams` → CVNN parameters
- `OptimizerState` → Adam state in `AdamOptimizerState`
- `RngTorchCpu` / `RngTorchCuda` / `RngNumpy` → serialized RNG state bytes
- `SobolSkip` / `GlobalStep` → metadata effects tracked in the interpreter
- `DeterminismFlags` → runtime determinism effect applied before execution

## Actions

Actions mirror the effect-sequenced workflow.

```tla
\* File: documents/engineering/tla.md
Init ==
  /\ DeterminismFlags = TRUE
  /\ GlobalStep = 0
  /\ SobolSkip = 0
  /\ EffectQueue = << >>

TrainStep ==
  /\ DeterminismFlags
  /\ NextModelParams
  /\ NextOptimizerState
  /\ NextRngStates
  /\ GlobalStep' = GlobalStep + 1
  /\ SobolSkip' = SobolSkip + 1

Snapshot ==
  /\ SnapshotStore' = SnapshotStore \cup {SnapshotState}

Restore ==
  /\ \E snap \in SnapshotStore: RestoreFrom(snap)
```

## Invariants

The proof centers on two invariants:

1. **Determinism**: same inputs imply same outputs.
2. **Resume equivalence**: snapshot → restore → N steps equals continuous N steps.

```tla
\* File: documents/engineering/tla.md
Deterministic ==
  \A s, s2 \in States:
    (s = s2) => (Step(s) = Step(s2))

ResumeEquivalence ==
  \A s \in States:
    TrainN(s, n) = TrainN(Restore(Snapshot(s)), n)
```

## Proof Strategy

- **Model checking (TLC)**: bounded checks for invariants on a reduced state
  space (small tensors, small step counts).
- **Inductive proofs (TLAPS)**: invariants proven over unbounded steps using
  explicit state invariants and action guards.

TLA+ proofs must reference the assumptions above and explicitly annotate any
abstraction boundary.

## Authoring Rules (Do This First)

These rules keep the spec aligned with SpectralMC doctrines and make TLC results
interpretable:

- **One spec per topic**: use a single `.tla` file per proof target.
- **Local config**: place the `.cfg` next to the `.tla` file.
- **Small state**: limit the model to small finite sets so TLC can terminate.
- **Module naming**: module name must match the `.tla` file name (case-sensitive).
- **Config binding**: put `SPECIFICATION` and `INVARIANT` in the `.cfg`, not in
  the `.tla` file.
- **Explicit variables**: every reproducibility-relevant field must appear in
  `VARIABLES` (RNG state, Sobol skip, global step, optimizer state).
- **No hidden effects**: model every effect as a state transition; never assume
  silent global state.
- **Match the effect boundary**: follow [effect_interpreter.md](effect_interpreter.md)
  and [total_pure_modelling.md](total_pure_modelling.md) for action structure.
- **No automation**: run TLC manually; do not add CI or git hooks.

## TLA+ Entrypoint Options

`poetry run tla-check` supports a minimal, explicit interface:

- `--spec <path>`: required `.tla` file path.
- `--config <path>`: optional `.cfg` file path (relative to the `.tla` directory).
- `--workers <n|auto>`: TLC worker count (default `auto`).
- `--tlc-arg <arg>`: pass through additional TLC arguments (repeatable).

## Guided Workflow (TLC)

All commands must run through Docker per repo policy. The examples below show
both the in-container command and the full `docker compose` wrapper.

### 1) Create the TLA+ module and config

Create a dedicated folder and two files (module + config):

```bash
# File: documents/engineering/tla.md
mkdir -p documents/engineering/tla
nano documents/engineering/tla/reproducibility.tla
nano documents/engineering/tla/reproducibility.cfg
```

**Minimal skeleton** (start here and expand):

```tla
\* File: documents/engineering/tla/reproducibility.tla
---- MODULE reproducibility ----
EXTENDS Naturals, Sequences

VARIABLES ModelParams, OptimizerState, RngTorchCpu, RngTorchCuda, RngNumpy,
          SobolSkip, GlobalStep, DeterminismFlags

Init ==
  /\ DeterminismFlags = TRUE
  /\ GlobalStep = 0
  /\ SobolSkip = 0

TrainStep ==
  /\ DeterminismFlags
  /\ GlobalStep' = GlobalStep + 1
  /\ SobolSkip' = SobolSkip + 1

Next == TrainStep

Deterministic == TRUE \* Placeholder, replace with real invariant

Spec == Init /\ [][Next]_<<ModelParams, OptimizerState, RngTorchCpu, RngTorchCuda, RngNumpy, SobolSkip, GlobalStep, DeterminismFlags>>

====
```

```tla
\* File: documents/engineering/tla/reproducibility.cfg
SPECIFICATION Spec
INVARIANT Deterministic
```

### 1.5) Keep the Model Small

TLC is a model checker, not a theorem prover. Reduce state size using small
sets and bounded counters.

```tla
\* File: documents/engineering/tla.md
CONSTANTS
  MaxStep <- 3
  ParamSet <- {0, 1}
```

### 2) Run TLC via Poetry

Use the new Poetry entrypoint to run TLC:

```bash
# File: documents/engineering/tla.md
# Run TLC (module path required; .tla extension is ok)
poetry run tla-check \
  --spec documents/engineering/tla/reproducibility.tla \
  --config documents/engineering/tla/reproducibility.cfg
```

```bash
# File: documents/engineering/tla.md
# Same command via docker compose
docker compose -f docker/docker-compose.yml exec spectralmc poetry run tla-check \
  --spec documents/engineering/tla/reproducibility.tla \
  --config documents/engineering/tla/reproducibility.cfg
```

### 3) Interpret the output

- **No errors**: TLC validated the invariant within the bounded model.
- **Counterexample**: TLC prints a trace. Update the model or assumptions.
- **Syntax error**: TLC prints the line and column; fix the TLA+ file.

## Common Errors (What To Fix)

- **Unknown operator**: check `EXTENDS` and module name casing.
- **Non-constant**: move values into `CONSTANTS` and bind them in the `.cfg`.
- **TLC out of memory**: reduce set sizes; shorten traces; simplify actions.
- **Invariant violated**: inspect the trace and add the missing state variable.

## Mapping to Code

The spec aligns with the reproducibility boundary documented in
[reproducibility_proofs.md](reproducibility_proofs.md). The following code
artifacts are directly modeled:

- `src/spectralmc/gbm_trainer.py` (train step + snapshot)
- `src/spectralmc/effects/interpreter.py` (RNG capture/restore, metadata effects)
- `src/spectralmc/runtime/torch_runtime.py` (deterministic runtime effect)

The TLA+ model is authoritative for reproducibility guarantees; tests provide
empirical validation only.

## Non-Goals

- Proving floating-point arithmetic identities across GPU architectures
- Proving determinism for non-deterministic library calls
- Modeling external services or storage backends beyond their effect contracts

## Manual Verification Workflow

- Write the TLA+ spec in `documents/engineering/tla/`.
- Run TLC manually with `poetry run tla-check`.
- Capture results in review notes; do not add automation or CI hooks.
