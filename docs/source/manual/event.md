(event)=

# Building events

Event building is the process of combining the information from various
subsystems. This also includes splitting simulated files into subsections
corresponding to different runs or periods in data taking.

## Time-coincidence map (TCM)

The basis of event building is the time-coincidence map or (TCM).

The TCM tells us which hits in the various subsystems happened close enough in
time to be considered part of the same "event".

The TCM is a {class}`lgdo.Table` with two fields, both of which are
{class}`lgdo.VectorOfVectors`:

- row_in_table: which row of the file contains this hit
- table_key: which channel was the hit in.

The mapping from table keys to detector names is contained in the links of the
`lh5` file. This can be converted to a python dictionary with the function
{func}`reboost.utils.get_remage_detector_uids`.

Since version 0.12.0 _remage_ can compute the TCM directly and store it in the
output file. After writing a hit tier file with
{func}`reboost.io.write_hit_table_chunk`, the TCM of the new file is built with
{func}`reboost.tcm.build_remage_tcm`, which uses the same coincidence settings
as _remage_.

## Gathering data from other fields

The first step of event building is to gather data from the various tiers. This
can be done with {func}`reboost.io.read_hit_field_by_tcm`. This will return the
data as a {class}`awkward.Array` with the same shape as the TCM.

From this more manipulation can be applied using awkward manipulations, or
custom written processors.

## Filtering channels

One useful functionality is to select groups of channels. To do this the
function {func}`reboost.shape.group.get_isin_group` can be used. This will
return an awkward array with the same shape as the channels input of booleans
indicating if a given channel was part of the group.
