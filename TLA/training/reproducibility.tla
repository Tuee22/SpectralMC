---- MODULE reproducibility ----
EXTENDS Naturals, Sequences

CONSTANT MaxSteps

VARIABLES
  ModelParams, OptimizerState,
  RngTorchCpu, RngTorchCuda, RngNumpy,
  SobolSkip, GlobalStep, DeterminismFlags,
  SnapshotStore

\* Expected deterministic state as a pure function of the global step
ExpectedModelParams(gs) == gs
ExpectedOptimizerState(gs) == gs
ExpectedRngTorchCpu(gs) == gs
ExpectedRngTorchCuda(gs) == gs
ExpectedRngNumpy(gs) == gs
ExpectedSobolSkip(gs) == gs

SnapshotFromState ==
  [ step |-> GlobalStep,
    model |-> ModelParams,
    optimizer |-> OptimizerState,
    sobol |-> SobolSkip,
    rng |-> [
      cpu   |-> RngTorchCpu,
      cuda  |-> RngTorchCuda,
      numpy |-> RngNumpy
    ]
  ]

Snapshots ==
  { [ step      |-> g,
      model     |-> ExpectedModelParams(g),
      optimizer |-> ExpectedOptimizerState(g),
      sobol     |-> ExpectedSobolSkip(g),
      rng       |-> [
        cpu   |-> ExpectedRngTorchCpu(g),
        cuda  |-> ExpectedRngTorchCuda(g),
        numpy |-> ExpectedRngNumpy(g)
      ]
    ] : g \in 0..MaxSteps
  }

Init ==
  /\ GlobalStep = 0
  /\ SobolSkip = 0
  /\ DeterminismFlags = TRUE
  /\ ModelParams = ExpectedModelParams(0)
  /\ OptimizerState = ExpectedOptimizerState(0)
  /\ RngTorchCpu = ExpectedRngTorchCpu(0)
  /\ RngTorchCuda = ExpectedRngTorchCuda(0)
  /\ RngNumpy = ExpectedRngNumpy(0)
  /\ SnapshotStore = {}

TrainStep ==
  /\ GlobalStep < MaxSteps
  /\ GlobalStep' = GlobalStep + 1
  /\ SobolSkip' = ExpectedSobolSkip(GlobalStep')
  /\ ModelParams' = ExpectedModelParams(GlobalStep')
  /\ OptimizerState' = ExpectedOptimizerState(GlobalStep')
  /\ RngTorchCpu' = ExpectedRngTorchCpu(GlobalStep')
  /\ RngTorchCuda' = ExpectedRngTorchCuda(GlobalStep')
  /\ RngNumpy' = ExpectedRngNumpy(GlobalStep')
  /\ SnapshotStore' = SnapshotStore
  /\ DeterminismFlags' = DeterminismFlags

SnapshotAction ==
  /\ SnapshotStore' = SnapshotStore \cup {SnapshotFromState}
  /\ UNCHANGED <<ModelParams, OptimizerState, RngTorchCpu, RngTorchCuda, RngNumpy, SobolSkip, GlobalStep, DeterminismFlags>>

RestoreAction ==
  /\ SnapshotStore # {}
  /\ \E snap \in SnapshotStore:
        /\ GlobalStep' = snap.step
        /\ ModelParams' = snap.model
        /\ OptimizerState' = snap.optimizer
        /\ SobolSkip' = snap.sobol
        /\ RngTorchCpu' = snap.rng.cpu
        /\ RngTorchCuda' = snap.rng.cuda
        /\ RngNumpy' = snap.rng.numpy
        /\ SnapshotStore' = SnapshotStore
        /\ DeterminismFlags' = DeterminismFlags

Skip == UNCHANGED <<
  ModelParams, OptimizerState,
  RngTorchCpu, RngTorchCuda, RngNumpy,
  SobolSkip, GlobalStep, DeterminismFlags,
  SnapshotStore
>>

Next == TrainStep \/ SnapshotAction \/ RestoreAction \/ Skip

Spec == Init /\ [][Next]_<<
  ModelParams, OptimizerState,
  RngTorchCpu, RngTorchCuda, RngNumpy,
  SobolSkip, GlobalStep, DeterminismFlags,
  SnapshotStore
>>

TypeOK ==
  /\ GlobalStep \in 0..MaxSteps
  /\ SobolSkip \in 0..MaxSteps
  /\ ModelParams \in 0..MaxSteps
  /\ OptimizerState \in 0..MaxSteps
  /\ RngTorchCpu \in 0..MaxSteps
  /\ RngTorchCuda \in 0..MaxSteps
  /\ RngNumpy \in 0..MaxSteps
  /\ DeterminismFlags \in BOOLEAN
  /\ SnapshotStore \subseteq Snapshots

DeterministicState ==
  /\ ModelParams = ExpectedModelParams(GlobalStep)
  /\ OptimizerState = ExpectedOptimizerState(GlobalStep)
  /\ RngTorchCpu = ExpectedRngTorchCpu(GlobalStep)
  /\ RngTorchCuda = ExpectedRngTorchCuda(GlobalStep)
  /\ RngNumpy = ExpectedRngNumpy(GlobalStep)
  /\ SobolSkip = ExpectedSobolSkip(GlobalStep)

ResumeEquivalence ==
  \A snap \in SnapshotStore:
    /\ snap.model = ExpectedModelParams(snap.step)
    /\ snap.optimizer = ExpectedOptimizerState(snap.step)
    /\ snap.sobol = ExpectedSobolSkip(snap.step)
    /\ snap.rng.cpu = ExpectedRngTorchCpu(snap.step)
    /\ snap.rng.cuda = ExpectedRngTorchCuda(snap.step)
    /\ snap.rng.numpy = ExpectedRngNumpy(snap.step)

====
