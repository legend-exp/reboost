# Introduction

_reboost_ is a Python library to post-process the output of
[_remage_](inv:remage#index), and more generally of Geant4, simulations. A
Geant4 simulation stops at the energy depositions in the detectors. _reboost_
provides the routines that apply the detector models on top of them: the
response of the HPGe surfaces, the charge collection and the pulse shape, the
scintillation light and its detection by SiPMs, the energy resolution. The
result is a set of quantities that can be compared with the measured data.

_reboost_ contains:

- a library of "processors", functions that compute these quantities hit by hit,
- functions to read the _remage_ output in chunks, apply processors and write
  the result to disk,
- a dedicated tool for computing and using scintillation optical maps.

_reboost_ is written for the
[default LH5 output files](inv:remage#manual-output) of _remage_. The
processors, however, only act on arrays and are general enough to be applied to
other data structures.

## Main concepts

```{glossary}
step
  A discrete energy deposition simulated by Geant4, with a position, a time, an
  energy and the particle that produced it. Steps are the raw output of
  _remage_, but not always one row per Geant4 step: _remage_ can merge nearby
  depositions and reassign the energy of low-energy tracks before writing them,
  and {mod}`reboost.shape.cluster` can group them further.

hit
  The group of steps that make up one physical interaction in a detector. Steps
  belong to the same hit when they occur in the same detector, in the same
  Geant4 event and within the time resolution of the detector. Each hit has a
  time `t0`, the time of its first step.

hit grouping
  The operation that groups steps into hits. _remage_ [does it by default](inv:remage#manual-output) and
  writes one table per detector with one row per hit, whose fields are lists of
  values, one per step. _reboost_ can also do it on files written without hit
  grouping, with the functions in {mod}`reboost.shape.group`.

detector table
  The table holding the hits of one detector, named after the detector in the
  geometry. _remage_ also links each table under its {term}`uid`.

uid
  The unique identifier of a detector, an integer assigned to each sensitive
  detector when registering it in the geometry. The output labels detectors by
  their uid, an integer being cheaper to store and to compare than a name.
  {func}`reboost.utils.get_remage_detector_uids` reads the mapping between uids
  and detector table names from a file.

time-coincidence map
TCM
  A table listing which hits of different detectors are in coincidence, that
  is, share the Geant4 event and are close in time. Each row points to the
  hits, by detector {term}`uid` and row in the detector table, that a real
  detector array would record together. It has the same structure as the TCM
  that _pygama_ builds for the measured data (see the [pygama
  manual](inv:pygama#/manuals/evt.rst*#tcm)). _remage_ writes it in its output,
  {func}`reboost.tcm.build_remage_tcm` rebuilds it for the files written by
  _reboost_.

processor
  A function that computes a new quantity for each hit from the fields of a
  detector table and a description of the detector. Processors do not change
  the number of hits. {ref}`processors` states the contract they follow.
```
