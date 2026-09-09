# Energy resolution

Whatever the detector, the readout smears the energy it measures. These
functions evaluate a resolution curve at a given energy and apply that smearing
to a table of hits, one set of parameters per channel. Nothing here is specific
to a detector type.

```{eval-rst}
.. autofunction:: reboost.math.get_resolution
.. autofunction:: reboost.math.apply_energy_resolution
.. autofunction:: reboost.math.gaussian_sample
```
