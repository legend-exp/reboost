# Clustering

_remage_ writes one row per Geant4 "step", a discrete energy deposition in a
detector. Before any physics processor runs, the steps have to be grouped into
"hits", the physical interactions the detector actually resolves in time. The
functions below build that structure and, optionally, split each hit into
clusters of nearby steps.

Grouping is the only operation in _reboost_ that changes the number of rows. See
{ref}`processors` for how it fits into the hit-tier processing.

## Grouping steps into hits

```{eval-rst}
.. autofunction:: reboost.shape.group_by_evtid
.. autofunction:: reboost.shape.group_by_time
```

## Clustering steps within a hit

```{eval-rst}
.. autofunction:: reboost.shape.apply_cluster
.. autofunction:: reboost.shape.cluster_by_step_length
.. autofunction:: reboost.shape.step_lengths
```
