# Pulse-shape observables

Pulse-shape discrimination separates single-site interactions from multi-site
ones by looking at the shape of the current pulse the detector produces. A full
pulse-shape simulation is expensive, so _reboost_ offers both cheap heuristics
built on the step positions and a template-based estimate of the current.

{doc}`../manual/psd` explains the physics behind these observables and how to
choose between them.

## Heuristics

```{eval-rst}
.. autofunction:: reboost.hpge.r90
.. autofunction:: reboost.hpge.drift_time_heuristic
```

## Drift time

```{eval-rst}
.. autofunction:: reboost.hpge.drift_time
```

## Current pulses and A/E

```{eval-rst}
.. autofunction:: reboost.hpge.maximum_current
.. autofunction:: reboost.hpge.get_current_waveform
.. autofunction:: reboost.hpge.get_current_template
```

## n+ surface response

Charge released in the n+ surface layer diffuses before it drifts, which slows
the rising edge of the current pulse. The surface response is computed once and
convolved with the bulk pulse.

```{eval-rst}
.. autofunction:: reboost.hpge.get_surface_response
.. autofunction:: reboost.hpge.get_surface_library
.. autofunction:: reboost.hpge.convolve_surface_response
.. autofunction:: reboost.hpge.make_convolved_surface_library
.. autofunction:: reboost.hpge.prepare_surface_inputs
```

## Field and pulse shape libraries

The drift time and the pulse shape at each point of the detector are read from
files precomputed on a grid in the cylindrical `(r, z)` plane.

```{eval-rst}
.. autofunction:: reboost.hpge.get_hpge_rz_field
.. autoclass:: reboost.hpge.HPGeRZField
   :members:
.. autofunction:: reboost.hpge.get_hpge_pulse_shape_library
.. autoclass:: reboost.hpge.HPGePulseShapeLibrary
   :members:
.. autofunction:: reboost.hpge.prepare_pulse_shape_library
```
