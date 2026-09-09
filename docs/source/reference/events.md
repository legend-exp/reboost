# Building events

Up to this point every detector is processed on its own. An event combines the
hits that happened close enough in time across detectors, which is what the
time-coincidence map (TCM) records. The TCM works in detector "uids", the
integer identifiers _remage_ assigns to each detector, so reading it back
usually means reading the name-to-uid mapping too.

{doc}`../manual/event` walks through building events from a set of hit files.

## Time-coincidence map

```{eval-rst}
.. autofunction:: reboost.build_remage_tcm
```

## Detector identifiers

```{eval-rst}
.. autofunction:: reboost.get_remage_detector_uids
```
