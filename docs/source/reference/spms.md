# Optical detectors

These processors take the energy deposited in the scintillator and produce the
photoelectrons an optical detector records: how many photons are emitted, how
many are detected, when they arrive, and with what pulse amplitude. The detector
enters only through its optical map and its response parameters, so the chain
does not depend on the kind of light sensor.

They are meant to be applied in this order. Each one is separately available so
that you can stop at the level of detail your analysis needs.

## From energy deposition to photoelectrons

```{eval-rst}
.. autofunction:: reboost.spms.emitted_scintillation_photons
.. autofunction:: reboost.spms.number_of_detected_photoelectrons
.. autofunction:: reboost.spms.detected_photoelectrons
.. autofunction:: reboost.spms.photoelectron_times
```

## Instrument response

```{eval-rst}
.. autofunction:: reboost.spms.cluster_photoelectrons
.. autofunction:: reboost.spms.smear_photoelectrons
.. autofunction:: reboost.spms.corrected_photoelectrons
```
