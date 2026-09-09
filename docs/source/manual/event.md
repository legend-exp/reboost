(event)=

# Building events

Event building combines what the different detectors recorded into events. It
also covers splitting a simulated file into the runs or the periods of a data
taking campaign.

## Time-coincidence map (TCM)

Event building starts from the time-coincidence map (TCM), the table that says
which hits happened close enough in time to be read out as one event.

The TCM is a {class}`~lgdo.types.table.Table` with two fields, both
{class}`~lgdo.types.vectorofvectors.VectorOfVectors` holding one list per event:

- `table_key`: the {term}`uid` of the detector each hit is in,
- `row_in_table`: the row of that detector table where the hit sits.

The links of the LH5 file map the uids to the detector table names.
{func}`reboost.utils.get_remage_detector_uids` reads them into a Python
dictionary.

_remage_ writes the TCM in its output by default, and it usually serves for the
hit tier too. Processors do not change the number of hits, so row `i` of a hit
table is the same hit as row `i` of the step table it was computed from, and the
map still points at the right rows.

It has to be rebuilt with {func}`reboost.tcm.build_remage_tcm` when this stops
holding:

- the hits are no longer the same: rows were dropped, reordered, or the steps
  were regrouped into different hits,
- detectors were added that _remage_ did not write, for instance the SiPM
  detectors that {ref}`applying an optical map <optics>` creates from the
  scintillator hits,
- several files were put together,
- a coincidence window other than the one used by _remage_ is wanted.

## Reading fields event by event

{func}`reboost.io.read_hit_field_by_tcm` reads one field from every detector
table and arranges it like the TCM: one list per event, one value per hit. The
result is an {class}`awkward.Array`, on which the usual _awkward_ operations and
your own processors apply.

## Selecting groups of detectors

{func}`reboost.shape.group.get_isin_group` says which hits belong to a given
group of detectors, the ones switched off in a run for example. It takes the
uids of the hits, a mapping from detector name to group and the mapping between
names and uids, and returns booleans with the shape of the uids given. Use them
as a mask on any field read through the TCM.
