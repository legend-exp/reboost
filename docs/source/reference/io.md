# Reading and writing files

A production run does not fit in memory, so the hit tier is written detector by
detector and in chunks. These functions create the output table, append rows to
it, and read a field back either for a range of events or arranged event by
event through the TCM.

```{eval-rst}
.. autofunction:: reboost.init_hit_table
.. autofunction:: reboost.write_hit_table_chunk
```

## Reading back

```{eval-rst}
.. autofunction:: reboost.get_rows_in_event_range
.. autofunction:: reboost.read_hit_field_by_tcm
```
