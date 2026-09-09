# Charge collection

The energy an HPGe detector reads out is not the energy deposited by the
particle: charge created close to the n+ surface is only partly collected. These
functions turn the energy deposited at each step into the charge actually
collected there.

The distance of each step to the surface is computed first, then converted into
an "activeness", the fraction of charge collected at that point. The energy is
then summed over the steps of a hit, and smeared with the
{doc}`energy resolution <resolution>` of the channel.

## Distance to the detector surface

```{eval-rst}
.. autofunction:: reboost.hpge.distance_to_surface
```

## Activeness models

```{eval-rst}
.. autofunction:: reboost.math.piecewise_linear_activeness
.. autofunction:: reboost.math.ex_lin_activeness
.. autofunction:: reboost.math.vectorised_active_energy
```
