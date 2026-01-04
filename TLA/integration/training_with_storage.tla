---- MODULE training_with_storage ----
EXTENDS Naturals, Sequences, FiniteSets

CONSTANTS MaxSteps, MaxElements, MaxManifests, MaxEffects

VARIABLES
  \* Training state
  ModelParams, OptimizerState,
  RngTorchCpu, RngTorchCuda, RngNumpy,
  SobolSkip, GlobalStep, DeterminismFlags,
  SnapshotStore,
  \* Storage state
  ElementStore, ManifestStore, HeadRef,
  \* Interpreter state
  EffectQueue, InterpreterState, EffectHistory

\* Domains
ElementIds == 0..MaxElements
ManifestIds == 1..MaxManifests
EffectDomain == 1..MaxEffects
NoManifest == 0
ContentDomain == ElementIds

HashOf(x) == x
SeqToSet(seq) == { seq[i] : i \in 1..Len(seq) }

ElementHash(content) == HashOf(content)
ManifestHash(id, elems, prev) == HashOf(<<id, elems, prev>>)

ElemRecord(eid, content) == [id |-> eid, content |-> content]
ManifestRecord(mid, elems, prev) == [
  id       |-> mid,
  elements |-> elems,
  prev     |-> prev,
  hash     |-> ManifestHash(mid, elems, prev)
]

EffectCommit(step, content) == [
  kind    |-> "Commit",
  step    |-> step,
  content |-> content
]

ElementIdsOf(store) == { e.id : e \in store }
ManifestIdsOf(store) == { m.id : m \in store }

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
  /\ ElementStore = {}
  /\ ManifestStore = {}
  /\ HeadRef = NoManifest
  /\ EffectQueue = << >>
  /\ InterpreterState = << >>
  /\ EffectHistory = << >>

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
  /\ UNCHANGED <<ElementStore, ManifestStore, HeadRef, EffectQueue, InterpreterState, EffectHistory>>

SnapshotAction ==
  /\ SnapshotStore' = SnapshotStore \cup {SnapshotFromState}
  /\ UNCHANGED <<
      ModelParams, OptimizerState, RngTorchCpu, RngTorchCuda, RngNumpy,
      SobolSkip, GlobalStep, DeterminismFlags,
      ElementStore, ManifestStore, HeadRef,
      EffectQueue, InterpreterState, EffectHistory
    >>

EmitCommitEffect ==
  /\ Len(EffectHistory) < MaxEffects
  /\ ModelParams \in ContentDomain
  /\ EffectQueue' = Append(EffectQueue, EffectCommit(GlobalStep, ModelParams))
  /\ EffectHistory' = Append(EffectHistory, EffectCommit(GlobalStep, ModelParams))
  /\ UNCHANGED <<
      ModelParams, OptimizerState, RngTorchCpu, RngTorchCuda, RngNumpy,
      SobolSkip, GlobalStep, DeterminismFlags, SnapshotStore,
      ElementStore, ManifestStore, HeadRef, InterpreterState
    >>

ProcessCommit ==
  /\ EffectQueue # << >>
  /\ HeadRef \in ManifestIdsOf(ManifestStore) \cup {NoManifest}
  /\ Head(EffectQueue).kind = "Commit"
  /\ LET eff == Head(EffectQueue) IN
       LET eid == ElementHash(eff.content) IN
         /\ eid \in ElementIds
         /\ \E newId \in ManifestIds \ ManifestIdsOf(ManifestStore):
              /\ newId > HeadRef
              /\ ManifestStore' = ManifestStore \cup {ManifestRecord(newId, {eid}, HeadRef)}
              /\ HeadRef' = newId
              /\ ElementStore' = IF eid \in ElementIdsOf(ElementStore)
                                 THEN ElementStore
                                 ELSE ElementStore \cup {ElemRecord(eid, eff.content)}
              /\ EffectQueue' = Tail(EffectQueue)
              /\ InterpreterState' = Append(InterpreterState, eff)
              /\ EffectHistory' = EffectHistory
              /\ UNCHANGED <<
                  ModelParams, OptimizerState, RngTorchCpu, RngTorchCuda, RngNumpy,
                  SobolSkip, GlobalStep, DeterminismFlags, SnapshotStore
                >>

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
        /\ UNCHANGED <<ElementStore, ManifestStore, HeadRef, EffectQueue, InterpreterState, EffectHistory>>

Skip == UNCHANGED <<
  ModelParams, OptimizerState,
  RngTorchCpu, RngTorchCuda, RngNumpy,
  SobolSkip, GlobalStep, DeterminismFlags,
  SnapshotStore, ElementStore, ManifestStore, HeadRef,
  EffectQueue, InterpreterState, EffectHistory
>>

Next ==
  TrainStep \/ SnapshotAction \/ EmitCommitEffect \/
  ProcessCommit \/ RestoreAction \/ Skip

Spec == Init /\ [][Next]_<<
  ModelParams, OptimizerState,
  RngTorchCpu, RngTorchCuda, RngNumpy,
  SobolSkip, GlobalStep, DeterminismFlags,
  SnapshotStore, ElementStore, ManifestStore, HeadRef,
  EffectQueue, InterpreterState, EffectHistory
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
  /\ \A e \in ElementStore:
        /\ e.id \in ElementIds
        /\ e.content \in ContentDomain
  /\ \A m \in ManifestStore:
        /\ m.id \in ManifestIds
        /\ m.elements \subseteq ElementIds
        /\ m.prev \in ManifestIds \cup {NoManifest}
  /\ HeadRef \in ManifestIds \cup {NoManifest}
  /\ EffectQueue \in Seq({ EffectCommit(s, c) : s \in 0..MaxSteps, c \in ContentDomain })
  /\ EffectHistory \in Seq({ EffectCommit(s, c) : s \in 0..MaxSteps, c \in ContentDomain })
  /\ InterpreterState \in Seq({ EffectCommit(s, c) : s \in 0..MaxSteps, c \in ContentDomain })

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

ChainIntegrity ==
  /\ \A m \in ManifestStore:
        /\ m.hash = ManifestHash(m.id, m.elements, m.prev)
        /\ m.prev = NoManifest \/ m.prev \in ManifestIdsOf(ManifestStore)
        /\ m.prev = NoManifest \/ m.prev < m.id
  /\ \A m \in ManifestStore: m.elements \subseteq ElementIdsOf(ElementStore)

HeadLinearity ==
  /\ HeadRef = NoManifest \/ HeadRef \in ManifestIdsOf(ManifestStore)

InterpreterOrder ==
  /\ InterpreterState = SubSeq(EffectHistory, 1, Len(InterpreterState))
  /\ EffectQueue = SubSeq(EffectHistory, Len(InterpreterState) + 1, Len(EffectHistory))

StorageConsistency ==
  /\ Cardinality(ElementIdsOf(ElementStore)) = Cardinality(ElementStore)
  /\ Cardinality(ManifestIdsOf(ManifestStore)) = Cardinality(ManifestStore)

====
