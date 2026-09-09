(processors)=

# Processors

A "processor" in _reboost_ is a function that computes a new quantity for each
hit, for example the energy deposited after correcting for the HPGe surface
response, or the number of photoelectrons detected by a SiPM. Processors are
plain Python functions. You apply them in a script that reads the _remage_
output, computes the quantities you need and writes them to disk.

## The input: hits

_remage_ groups the "steps" (the discrete energy depositions of Geant4) into
"hits", one per physical interaction in a detector, and writes one table per
detector with one row per hit (see
[the remage manual](inv:remage#manual-output)). The fields of a hit are lists of
values, one per step:

```text
evtid  t0    edep              xloc              time
0      12.3  [1.2, 0.4, 5.1]   [0.11, 0.11, ...] [12.3, 12.4, 13.9]
0      98.0  [0.1]             [0.03]            [98.0]
2      3.5   [300.0, 2.2]      [-0.04, -0.04]    [3.5, 3.5]
```

Loaded with {func}`~lh5.io.core.read` and viewed as an {class}`awkward.Array`,
this is a jagged array with one entry per hit and variable-length lists inside.
Processors act on this structure.

## Using processors

Every processor takes the fields it needs as separate arguments and returns the
new quantity as an {class}`awkward.Array` with one entry per hit. For example,
the active energy of a HPGe detector:

```python
import awkward as ak
import lh5
import reboost

steps = lh5.read("stp/det001", "stp.lh5").view_as("ak", with_units=True)

# distance of each step to the n+ surface, in mm
dist = reboost.hpge.distance_to_surface(
    steps.xloc, steps.yloc, steps.zloc, hpge, det_pos, surface_type="nplus"
)
# fraction of charge collected at each step
activeness = reboost.math.piecewise_linear_activeness(dist, fccd_in_mm=1.0, dlf=0.5)
# active energy of each hit, in keV
energy = ak.sum(steps.edep * activeness, axis=-1)
```

Processors compose through _awkward_ operations: {func}`awkward.sum` over the
innermost axis reduces a per-step quantity to a per-hit one. The output of one
processor is the input of the next.

The available processors are documented in the
{doc}`API reference <../api/modules>`, grouped by subsystem: {mod}`reboost.hpge`
for germanium detectors, {mod}`reboost.spms` for SiPMs, {mod}`reboost.math` for
generic operations such as energy smearing, and {mod}`reboost.shape` for
reshaping (clustering steps, grouping).

To write the result to disk, see {func}`reboost.io.init_hit_table` and
{func}`reboost.io.write_hit_table_chunk`. Large files are processed in chunks
with {class}`~lh5.io.iterator.LH5Iterator`, applying the processors to each
chunk.

(processors-contract)=

## The processor contract

You can write your own processors. The only prescription for a processor is that
it returns an {class}`awkward.Array` with the same length as its inputs: a
processor acts on every hit, it cannot add, remove or merge hits. It can change
the inner structure: sum over steps (a "reduction"), split the steps of a hit
into clusters (adding a dimension), or compute a new per-step quantity.

The processors in _reboost_ follow these conventions. Your own processors should
too, so that they compose with the others.

- Data fields are the first arguments, one per field, as {class}`awkward.Array`
  or as {class}`~lgdo.types.lgdo.LGDO` objects. Pass fields, not the whole
  table.
- Objects describing the detector come next: a {class}`pygeomhpges.base.HPGe`
  instance, an {class}`optical map <reboost.optmap.convolve.OptmapForConvolve>`,
  a {class}`drift-time map <reboost.hpge.utils.HPGeRZField>`, the detector
  position.
- Settings are keyword arguments with defaults.
- Physical units travel with the data as attributes, following the
  [LEGEND data format](https://legend-exp.github.io/legend-data-format-specs/dev/hdf5/#Values-with-physical-units).
  A processor converts its inputs to the units it computes with, and documents
  the units of its output. The {mod}`reboost.units` module has the helpers:
  {func}`reboost.units.units_conv_ak` converts an array to given units, whatever
  units it carries, and {func}`reboost.units.attach_units` labels the output.

A minimal processor computing the energy-weighted mean height of a hit:

```python
import awkward as ak
from reboost import units


def mean_height(edep: ak.Array, zloc: ak.Array) -> ak.Array:
    """Energy-weighted mean of the step heights, in mm."""
    edep = units.units_conv_ak(edep, "keV")
    zloc = units.units_conv_ak(zloc, "mm")

    out = ak.sum(edep * zloc, axis=-1) / ak.sum(edep, axis=-1)

    return units.attach_units(out, "mm")
```

:::{note}

Step grouping. Files written by _remage_ with hit grouping disabled contain one
row per step instead of one per hit. The processors in
{mod}`reboost.shape.group` build the hit structure from such files:
{func}`reboost.shape.group.group_by_evtid` groups the steps of each Geant4
event, {func}`reboost.shape.group.group_by_time` also splits an event when the
time between steps exceeds a window (10 us by default). These are the only
processors that change the number of rows.

```python
steps = lh5.read("stp/det001", "stp_flat.lh5").view_as("ak")
hits = reboost.shape.group_by_time(steps, window=10)  # unit is us
```

:::
