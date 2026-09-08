from __future__ import annotations

import logging
from pathlib import Path

import awkward as ak
import h5py
import lh5
import numpy as np
from lgdo import LGDO, Array, Table

from .utils import get_remage_detector_uids

log = logging.getLogger(__name__)


def init_hit_table(stp_table: Table) -> Table:
    """Create a hit table carrying the event fields of a step table.

    The returned table has the same number of rows as `stp_table` and two
    fields, ``evtid`` (the Geant4 event identifier) and ``t0`` (the time of the
    hit, in nanoseconds). These are the fields that
    :func:`reboost.tcm.build_remage_tcm` needs to group hits of different
    detectors into events. Attach the processor outputs to this table with
    :meth:`~lgdo.types.table.Table.add_field` and write it with
    :func:`write_hit_table_chunk`.

    We assume that `stp_table` has one row per hit, a list of steps, as in the
    remage output with hit grouping enabled. If ``t0`` is missing, the time of
    the first step of each hit is used. If ``evtid`` is stored per step, the
    event identifier of the first step is used. The times must be in
    nanoseconds.
    """
    out = Table(size=len(stp_table))

    if "t0" in stp_table and isinstance(stp_table.t0, Array):
        t0 = stp_table.t0
    else:
        t0 = Array(
            ak.fill_none(ak.firsts(stp_table.time.view_as("ak"), axis=-1), 0),
            attrs={"units": stp_table.time.attrs.get("units", "ns")},
        )

    if isinstance(stp_table.evtid, Array):
        evtid = stp_table.evtid
    else:
        evtid = Array(ak.fill_none(ak.firsts(stp_table.evtid.view_as("ak"), axis=-1), 0))

    # the TCM building assumes nanoseconds
    units = t0.attrs.get("units")
    if units is None:
        log.warning("t0 has no units, assuming nanoseconds")
    elif units != "ns":
        msg = f"t0 must be in nanoseconds, found units '{units}'"
        raise ValueError(msg)

    out.add_field("t0", t0)
    out.add_field("evtid", evtid)

    return out


def write_hit_table_chunk(
    table: Table, name: str, file: str | Path, *, uid: int | None = None
) -> None:
    """Append rows of a detector table of a hit-like tier to an LH5 file.

    The first call for a given `name` creates the table, the file and the tier
    group when missing. Later calls append rows to the table. Every chunk must
    have the same fields. With `uid` set, the first call also registers the
    table in the ``__by_uid__`` group of the tier as a soft link named
    ``det<uid>``, with the identifier zero-padded to three digits (``det001``).
    This mirrors the layout of the remage output, so that
    :func:`reboost.utils.get_remage_detector_uids` and
    :func:`reboost.tcm.build_remage_tcm` work on the file.

    Warning
    -------
    The function appends to whatever it finds in the file, so it cannot tell
    the first chunk of a run from a table left by an earlier, failed run.
    Remove the output file before the first call.

    Parameters
    ----------
    table
        the rows to write.
    name
        name of the table in the file, e.g. ``hit/det001``. The last component
        is the table name, the rest is the tier group.
    file
        path to the output file, created if missing.
    uid
        remage unique identifier of the detector. If ``None``, no soft link is
        created.
    """
    file = str(file)
    name = name.strip("/")
    group, _, table_name = name.rpartition("/")

    if group == "":
        msg = f"the table name must contain the tier group, e.g. 'hit/{name}'"
        raise ValueError(msg)

    wo_mode = "append" if _exists(file, name) else "append_column"

    lh5.write(table, name, file, wo_mode=wo_mode)

    if uid is not None and wo_mode == "append_column":
        _add_uid_link(file, group, table_name, uid)


def _exists(file: str, name: str) -> bool:
    if not Path(file).is_file():
        return False

    # no file locking: some filesystems do not support it (e.g. CFS at NERSC)
    with h5py.File(file, "r", locking=False) as f:
        return name in f


def _add_uid_link(file: str, group: str, table_name: str, uid: int) -> None:
    """Register `table_name` in the ``__by_uid__`` group of `group`.

    The group is created if missing. Unlike the tables, it is not listed among
    the fields of the tier struct, so that reading the tier group does not
    follow the links.
    """
    link_name = f"det{uid:03}"
    target = f"/{group}/{table_name}"

    log.debug("creating soft link %s/__by_uid__/%s -> %s", group, link_name, target)

    # no file locking: some filesystems do not support it (e.g. CFS at NERSC)
    with h5py.File(file, "r+", locking=False) as f:
        g = f.require_group(f"{group}/__by_uid__")

        if link_name in g:
            msg = f"uid {uid} is already registered in {file}:{group}/__by_uid__"
            raise ValueError(msg)

        g[link_name] = h5py.SoftLink(target)

        # update the struct datatype attribute by adding the new link
        fields = lh5.io.datatype.get_struct_fields(g.attrs.get("datatype", "struct{}"))
        g.attrs["datatype"] = "struct{" + ",".join(sorted([*fields, link_name])) + "}"


def get_rows_in_event_range(
    tcm: ak.Array | Table, uid: int, first: int, last: int
) -> tuple[int, int]:
    """Find the rows of a detector table that belong to a range of events.

    Each row of the time-coincidence map (TCM) is an event, a group of hits
    close in time. This function selects the events with index between `first`
    and `last` (both included) and returns the index of the first row of the
    table with unique identifier `uid` that belongs to them, together with the
    number of rows. The result can be passed to
    :class:`~lh5.io.iterator.LH5Iterator` as ``i_start`` and
    ``n_entries``. Use it to process only part of a file, for example to split
    a simulation into ranges of events assigned to different data-taking runs.

    We assume that the rows of the detector table are stored in event order, as
    remage does, so that the rows belonging to a range of events are
    contiguous.

    Parameters
    ----------
    tcm
        the time-coincidence map, with fields ``table_key`` and
        ``row_in_table``.
    uid
        remage unique identifier of the detector.
    first
        index of the first event of the range.
    last
        index of the last event of the range (included). Truncated with a
        warning if past the end of the TCM.

    Returns
    -------
    the index of the first row and the number of rows. If no hit was recorded
    in the detector for these events, ``(0, 0)``.

    Examples
    --------
    >>> i_start, n_entries = get_rows_in_event_range(tcm, uid, 0, 999)
    >>> it = lh5.LH5Iterator(file, f"stp/{name}", i_start=i_start, n_entries=n_entries)
    """
    if isinstance(tcm, LGDO):
        tcm = tcm.view_as("ak")

    if first < 0 or last < 0:
        msg = "only non-negative event indices are supported"
        raise ValueError(msg)

    if last < first:
        msg = f"the last event index ({last}) is smaller than the first ({first})"
        raise ValueError(msg)

    if last >= len(tcm):
        log.warning(
            "the last event index (%d) is past the end of the TCM (%d events), truncating",
            last,
            len(tcm),
        )
        last = len(tcm) - 1

    # add one for inclusive slicing
    tcm_part = tcm[first : last + 1]

    rows = ak.flatten(tcm_part.row_in_table[tcm_part.table_key == uid]).to_numpy()

    if len(rows) == 0:
        log.debug("no hits recorded in table with uid %d in events [%d, %d]", uid, first, last)
        return 0, 0

    if np.any(np.diff(rows) != 1):
        msg = f"the rows of table with uid {uid} are not contiguous in events [{first}, {last}]"
        raise ValueError(msg)

    log.debug(
        "hits with rows in [%d, %d] recorded in table with uid %d in events [%d, %d]",
        rows[0],
        rows[-1],
        uid,
        first,
        last,
    )

    return int(rows[0]), len(rows)


def read_hit_field_by_tcm(
    tcm: ak.Array | Table,
    file: str | Path,
    field: str,
    uids: dict[int, str] | None = None,
    *,
    lh5_group: str = "hit",
    with_units: bool = False,
) -> ak.Array:
    """Read a field of the detector tables of a hit-like tier, arranged by event.

    A hit-like tier stores one table per detector, with one row per hit. The
    time-coincidence map (TCM) lists, for each event, the detectors that
    recorded a hit (``table_key``) and the row of the hit in the detector table
    (``row_in_table``). This function reads `field` from each detector table at
    those rows and returns it with the shape of the TCM: one list per event,
    one value per hit, in the order of the TCM.

    Parameters
    ----------
    tcm
        the time-coincidence map, or a slice of it.
    file
        the hit-like tier file.
    field
        name of the field, relative to the detector table. Nested fields are
        reached with a slash, e.g. ``psd/aoe``.
    uids
        mapping between the remage unique identifiers in ``table_key`` and the
        detector table names, as returned by
        :func:`reboost.utils.get_remage_detector_uids`. Every detector in the
        TCM must be present. Read from the file links if ``None``, pass it
        explicitly when reading many fields.
    lh5_group
        the tier group holding the detector tables.
    with_units
        attach the units of the field, if any, as the ``units`` parameter of
        the output array (see :mod:`reboost.units`).

    Examples
    --------
    >>> tcm = lh5.read_as("tcm", "hit.lh5", "ak")
    >>> uids = get_remage_detector_uids("hit.lh5", lh5_table="hit")
    >>> energy = read_hit_field_by_tcm(tcm, "hit.lh5", "energy", uids)
    >>> aoe = read_hit_field_by_tcm(tcm, "hit.lh5", "psd/aoe", uids)
    """
    if isinstance(tcm, LGDO):
        tcm = tcm.view_as("ak")

    if uids is None:
        uids = get_remage_detector_uids(file, lh5_table=lh5_group)

    file = str(file)

    # for un-flattening at the end
    counts = ak.num(tcm.row_in_table)

    table_key = ak.flatten(tcm.table_key).to_numpy()
    row_in_table = ak.flatten(tcm.row_in_table).to_numpy()

    data = []
    positions = []
    units = None

    for uid, name in uids.items():
        mask = table_key == uid
        if not np.any(mask):
            continue

        # the position of the hits of this detector in the flattened TCM
        positions.append(np.where(mask)[0])

        # read the rows in ascending order. remage writes hits in event order,
        # so this is a contiguous range and we can avoid a slow indexed read
        rows = row_in_table[mask]
        order = np.argsort(rows)
        rows = rows[order]

        lh5_name = f"{lh5_group}/{name}/{field}"
        if np.all(np.diff(rows) == 1):
            obj = lh5.read(lh5_name, file, start_row=int(rows[0]), n_rows=len(rows))
        else:
            log.debug("hits of %s are not contiguous, reading with an index list", name)
            obj = lh5.read(lh5_name, file, idx=rows)

        if len(obj) != len(rows):
            msg = f"read {len(obj)} rows from {file}:{lh5_name} instead of {len(rows)}"
            raise ValueError(msg)

        units = obj.attrs.get("units", units)

        # back to the order of the TCM
        data.append(obj.view_as("ak")[np.argsort(order)])

    n_read = sum(len(p) for p in positions)
    if n_read != len(table_key):
        msg = f"{len(table_key) - n_read} hits in the TCM have no table in `uids`"
        raise ValueError(msg)

    if not data:
        return ak.unflatten(ak.Array([]), counts)

    data = ak.concatenate(data)[np.argsort(np.concatenate(positions))]
    data = ak.unflatten(data, counts)

    if with_units and units is not None:
        return ak.with_parameter(data, "units", units)

    return data
