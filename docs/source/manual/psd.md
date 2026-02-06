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
HPGe detector. See [link](https://github.com/legend-exp/legend-simflow/blob/main/workflow/src/legendsimflow/scripts/make_hpge_drift_time_maps.jl) for example.

_reboost_ defines a common input format for these mappings as described in {func}`reboost.hpge.utils.get_rz_field`.

Alternatively for a more complete description of the PSD parameters a library of simulated waveforms can be used, again _reboost_
defines a common input format (see {func}`reboost.hpge.utils.get_hpge_pulse_shape_library`).

## Drift time heuristic

From this the drift time for each interaction in the HPGe detector can be extracted with {func}`reboost.hpge.psd.drift_time`. Based on this it is possible to compute the "drift time heuristic" (see {func}`reboost.hpge.psd.drift_time_heuristic`). This is an arbitrary value describing the likelihood of an event to fail a PSD cut. When compared to calibration data this can be used to emulate the effect of PSD cuts.

## A/E estimation

Additionally _rebost_ contains experimental methods to estimate the A/E parameter used for PSD in point-contact HPGe detectors.

These models are based on combining the "impulse response" (waveform) of each step in the Geant4 event with time-shifts given by the drift time maps. This can either be based on a simple constant current waveform model (described in {func}`reboost.hpge.psd._current_pulse_model`) or a library of simulated pulse shapes ({func}`reboost.hpge.utils.get_hpge_pulse_shape_library`).

In both cases the maximum current ("A" from A/E) can be estimated by {func}`reboost.hpge.psd.maximum_current`.

The following lines of code allow to extract the drift time for each simulated hit, and estimate the A/E (based on a single template).

```python
# extract the necessary inputs
template, times = reboost.hpge.psd.get_current_waveforms(...)
drift_time_map = reboost.hpge.utils.get_rz_field(...)

# extract drift time
drift_time = reboost.hpge.psd.drift_time(xloc, yloc, zloc, dt_map, ...)

a_max = reboost.hpge.psd.maximum_current(
    edep, drift_time, template=template, times=times
)

# finally compute a/e
aoe = a_max / ak.sum(edep, axis=-1)
```

This function is highly optimised by only extracting the waveform value close to the maximum. This combined with the fact the computionally intensive charge drift simulation is done in advance means this is orders of magnitude faster than a full charge drift simulation.

## n+ surface effects

Finally, _reboost_ contains methods to account for the delayed charge collection in the lithiated n+ contact of HPGe detectors. This follows an approach similar to [link](https://arxiv.org/abs/1605.03978).

The "surface response" of the detector, or the amount of charge arriving at the p-n junction as a function of time can be computed with {func}`reboost.hpge.surface.get_surface_response`. A matrix describing the response as a function of time and depth to the surface can be extracted with {func}`reboost.hpge.surface.get_surface_library`. The value in the final column of each row gives the charge collection efficiency (or "activeness").

This can then also be used in the determination of A/E using {func}`reboost.hpge.psd.maximum_current`.

:::{note}

Currently n+ effects can only be included with the fixed template option of {func}`reboost.hpge.psd.maximum_current`.

:::

## Waveform simulation

The library of simulated pulse shapes can also be used to extract a simulated waveform per event (see {func}`reboost.hpge.psd.waveform_pulse_shape_library`). This is simply performed by summing the waveforms in the pulse shape library weighted by their
energy.

The following lines of code allow to extract a waveform per event:

```python
# create the library
library = reboost.hpge.utils.get_hpge_pulse_shape_library(...)

# read the remage file
steps = lh5.read("stp", ...).view_as("ak", with_units=True)

# compute r, here x0, y0, z0 are the detector origin
r = np.sqrt((steps.xloc - x0) ** 2 + (steps.yloc - y0) ** 2)

# now compute waveforms
waveforms = reboost.hpge.psd.waveform_from_pulse_shape_library(
    steps.edep, r, (steps.zloc - z0), library
)

# now plot etc
```

:::{warning}

This is many times slower than the {func}`reboost.hpge.psd.maximum_current` since the value of the waveform at every sample is calculated. The large size of the output waveforms means the function can also use a significant amount of memory.
:::
