---- MODULE effects ----
EXTENDS Naturals, Sequences

\* Effect queue placeholder.
Effect == Nat

EmptyQueue == << >>

Enqueue(queue, eff) == Append(queue, eff)

====
