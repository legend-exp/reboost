# Optical maps

Tracking every scintillation photon through the detector is far too slow for
production simulations. Instead, _reboost_ precomputes an "optical map", the
probability that a photon emitted at a given point reaches each light detector,
and applies it to the energy depositions afterwards.

{doc}`../manual/optical` describes how to produce a map from a _remage_
simulation and how to apply it. The command line tool `reboost-optical` covers
the usual workflow.

```{eval-rst}
.. autoclass:: reboost.optmap.OpticalMap
   :members:
```

## Loading a map for use in processors

```{eval-rst}
.. autofunction:: reboost.spms.load_optmap
.. autofunction:: reboost.spms.load_optmap_all
```
