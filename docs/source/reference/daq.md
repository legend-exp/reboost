# DAQ emulation

A real data acquisition system does not record every event it is offered. It has
a finite readout time and can be busy when the next event arrives. This
processor pipes simulated events through that behavior so that the simulated
rates can be compared with the measured ones.

```{eval-rst}
.. autofunction:: reboost.daq.run_daq_non_sparse
```
