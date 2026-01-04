---- MODULE effect_interpreter ----
EXTENDS Naturals, Sequences

CONSTANT MaxEffects

VARIABLES EffectQueue, InterpreterState, EffectHistory

EffectDomain == 1..MaxEffects
EmptyQueue == << >>

Init ==
  /\ EffectQueue = EmptyQueue
  /\ InterpreterState = EmptyQueue
  /\ EffectHistory = EmptyQueue

Enqueue ==
  /\ Len(EffectHistory) < MaxEffects
  /\ \E eff \in EffectDomain:
        /\ EffectQueue' = Append(EffectQueue, eff)
        /\ EffectHistory' = Append(EffectHistory, eff)
        /\ InterpreterState' = InterpreterState

ProcessEffect ==
  /\ EffectQueue # EmptyQueue
  /\ LET eff == Head(EffectQueue) IN
       /\ EffectQueue' = Tail(EffectQueue)
       /\ InterpreterState' = Append(InterpreterState, eff)
       /\ EffectHistory' = EffectHistory

Skip == UNCHANGED <<EffectQueue, InterpreterState, EffectHistory>>

Next == Enqueue \/ ProcessEffect \/ Skip

Spec == Init /\ [][Next]_<<EffectQueue, InterpreterState, EffectHistory>>

TypeOK ==
  /\ EffectQueue \in Seq(EffectDomain)
  /\ InterpreterState \in Seq(EffectDomain)
  /\ EffectHistory \in Seq(EffectDomain)

ProcessedIsPrefix ==
  InterpreterState = SubSeq(EffectHistory, 1, Len(InterpreterState))

QueueFollowsHistory ==
  EffectQueue = SubSeq(EffectHistory, Len(InterpreterState) + 1, Len(EffectHistory))

====
