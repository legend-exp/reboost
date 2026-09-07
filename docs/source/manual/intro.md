# Introduction

_reboost_ contains both:

- a library of useful functions for applying post-processing steps "processors",
- a dedicated tool for computing and using scintillation optical maps.

{ref}`processors` describes _reboost_ processors and how to use these for a
simple simulation post-processing in a python script. Finally, the information
from multiple systems can be combined to build events (described in
{ref}`event`).

## Tiers in _reboost_

The simulation workflow in remage/reboost is divided into several "tiers", in a
similar way to the pygama data processing. This is then reflected in the name of
the lh5 group for each file. Currently we have the following tiers, these mirror
those in pygama:

- **stp**: The "step" information from remage / geant4. More information can be
  found in the remage documentation!
- **hit**: The data after grouping steps according to the physical interaction
  time-scale in the detector, and applying simulation post-processing steps that
  depend on only a single detector.
- **tcm**: Or "time-coincidence-map" a mapping of which hits happened
  simultaneously (within the detector time resolution) between different
  detectors.
- **evt**: Or "event" the final output file combining the information from the
  various detectors/ subsystems.

The next sections describe the **hit** tier processing.
