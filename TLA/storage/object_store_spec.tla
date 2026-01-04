---- MODULE object_store_spec ----
EXTENDS Naturals, Sequences, FiniteSets

CONSTANTS MaxElements, MaxManifests

VARIABLES ElementStore, ManifestStore, HeadRef

\* Domain definitions
ElementIds == 0..MaxElements
ManifestIds == 1..MaxManifests
NoManifest == 0
ContentDomain == ElementIds

HashOf(x) == x
ElementHash(content) == HashOf(content)
ManifestHash(id, elems, prev) == HashOf(<<id, elems, prev>>)

ElemRecord(eid, content) == [id |-> eid, content |-> content]
ManifestRecord(mid, elems, prev) == [
  id       |-> mid,
  elements |-> elems,
  prev     |-> prev,
  hash     |-> ManifestHash(mid, elems, prev)
]

ElementIdsOf(store) == { e.id : e \in store }
ManifestIdsOf(store) == { m.id : m \in store }

Init ==
  /\ ElementStore = {}
  /\ ManifestStore = {}
  /\ HeadRef = NoManifest

WriteElement ==
  \E content \in ContentDomain:
    LET eid == ElementHash(content) IN
      /\ eid \in ElementIds
      /\ IF eid \in ElementIdsOf(ElementStore)
         THEN ElementStore' = ElementStore
         ELSE ElementStore' = ElementStore \cup {ElemRecord(eid, content)}
      /\ UNCHANGED <<ManifestStore, HeadRef>>

CommitManifest ==
  /\ HeadRef \in ManifestIdsOf(ManifestStore) \cup {NoManifest}
  \* Choose a fresh manifest id and append to the chain
  /\ \E newId \in ManifestIds \ ManifestIdsOf(ManifestStore):
        \E elems \in SUBSET ElementIdsOf(ElementStore):
          /\ elems # {}
          /\ newId > HeadRef
          /\ ManifestStore' = ManifestStore \cup {ManifestRecord(newId, elems, HeadRef)}
          /\ HeadRef' = newId
          /\ ElementStore' = ElementStore

Skip == UNCHANGED <<ElementStore, ManifestStore, HeadRef>>

Next == WriteElement \/ CommitManifest \/ Skip

Spec == Init /\ [][Next]_<<ElementStore, ManifestStore, HeadRef>>

TypeOK ==
  /\ \A e \in ElementStore:
        /\ e.id \in ElementIds
        /\ e.content \in ContentDomain
  /\ \A m \in ManifestStore:
        /\ m.id \in ManifestIds
        /\ m.elements \subseteq ElementIds
        /\ m.prev \in ManifestIds \cup {NoManifest}
  /\ HeadRef \in ManifestIds \cup {NoManifest}

UniqueElements == Cardinality(ElementIdsOf(ElementStore)) = Cardinality(ElementStore)
UniqueManifests == Cardinality(ManifestIdsOf(ManifestStore)) = Cardinality(ManifestStore)

ChainIntegrity ==
  /\ UniqueElements
  /\ UniqueManifests
  /\ \A m \in ManifestStore:
        /\ m.hash = ManifestHash(m.id, m.elements, m.prev)
        /\ m.prev = NoManifest \/ m.prev \in ManifestIdsOf(ManifestStore)
        /\ m.prev = NoManifest \/ m.prev < m.id

HeadLinearity ==
  /\ HeadRef = NoManifest \/ HeadRef \in ManifestIdsOf(ManifestStore)
  /\ \A m \in ManifestStore: m.elements \subseteq ElementIdsOf(ElementStore)

====
