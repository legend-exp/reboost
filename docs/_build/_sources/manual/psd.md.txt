(psd)=

# Simulations of HPGe pulse-shape observables

_reboost_ contains a series of functionality to simulate pulse
shape observables for HPGe detectors.

:::{warning}
These simulation methods are experimental and should be employed at the users own risk!

:::

## Drift time maps

Charge drift software such as [SolidStateDetectors.jl](https://juliaphysics.github.io/SolidStateDetectors.jl/stable/),
can be used to compute the "drift time", or the time for charges to drift until reaching the contact, for each point in the
HPGe detector.

_reboost_ defines a common input format for these mappings as described in {func}`reboost.hpge.utils.get_rz_field`.

## Drift time heuristic

From this the drift time for each interaction in the HPGe detector can be extracted with {func}`reboost.hpge.psd.drift_time`. Based on this it is possible to compute the "drift time heuristic" (see {func}`reboost.hpge.psd.drift_time_heuristic`). This is an arbitrary value describing the likelihood of an event to fail a PSD cut. When compared to calibration data this can be used to emulate the effect of PSD cuts.

## A/E estimation

Additionally _rebost_ contains experimental methods to estimate the A/E parameter used for PSD in point-contact HPGe detectors. This can either be based on a simple constant current waveform model (described in {func}`reboost.hpge.psd._current_pulse_model`). In this case the maximum current ("A" from A/E) can be estimated by {func}`reboost.hpge.psd.maximum_current`.
The following lines of code allow to extract the drift time for each simulated hit, and estimate the A/E.

```python
# extract the necessary inputs
template, times = reboost.hpge.psd.get_hpge_pulse_shape_library(...)
drift_time_map = reboost.hpge.utils.get_rz_field(...)

# extract drift time
drift_time = reboost.hpge.psd.drift_time(xloc, yloc, zloc, dt_map, ...)

a_max = reboost.hpge.psd.maximum_current(
    edep, drift_time, template=template, times=times
)

# finally compute a/e
aoe = a_max / ak.sum(edep, axis=-1)
```

Or instead a template per point of the HPGe detector can be employed, _reboost_ employs a similar input file format as described in {func}`reboost.hpge.utils.get_hpge_pulse_shape_library`. In this case this library can be passed to {func}`reboost.hpge.psd.maximum_current`.

## n+ surface effects

Finally, _reboost_ contains methods to account for the delayed charge collection in the lithiated n+ contact of HPGe detectors. This follows an approach similar to [link](https://arxiv.org/abs/1605.03978).

The "surface response" of the detector, or the amount of charge arriving at the p-n junction as a function of time can be computed with {func}`reboost.hpge.surface.get_surface_response`. A matrix describing the response as a function of time and depth to the surface can be extracted with {func}`reboost.hpge.surface.get_surface_library`. The value in the final column of each row gives the charge collection efficiency (or "activeness").

This can then also be used in the determination of A/E using {func}`reboost.hpge.psd.maximum_current`.
