# Physical units

Every quantity that _reboost_ passes around carries its physical units, either
as a {class}`pint.Quantity` or as the `units` attribute of an
{class}`awkward.Array` or {class}`~lgdo.types.lgdo.LGDO`, following the
[LEGEND data format specification](https://legend-exp.github.io/legend-data-format-specs/dev/hdf5/#Values-with-physical-units).
Processors convert their inputs to the units they work in, so you do not have to
remember whether a drift time map is in nanoseconds or microseconds.

These functions attach, read and convert those units.

```{eval-rst}
.. autodata:: reboost.units.ureg
   :no-value:
   :no-index:
```

## Attaching and reading units

```{eval-rst}
.. autofunction:: reboost.units.attach_units
   :no-index:
.. autofunction:: reboost.units.get_unit_str
   :no-index:
.. autofunction:: reboost.units.unwrap_lgdo
   :no-index:
```

## Converting

```{eval-rst}
.. autofunction:: reboost.units.units_conv_ak
   :no-index:
.. autofunction:: reboost.units.units_convfact
   :no-index:
.. autofunction:: reboost.units.move_units_to_flattened_data
   :no-index:
```

## Interfacing with other packages

```{eval-rst}
.. autofunction:: reboost.units.unit_to_lh5_attr
   :no-index:
.. autofunction:: reboost.units.pg4_to_pint
   :no-index:
```
