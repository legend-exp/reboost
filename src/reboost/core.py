from __future__ import annotations

import logging

import awkward as ak
import lh5
import numpy as np

log = logging.getLogger(__name__)


def read_data_at_channel_as_ak(
    channels: ak.Array,
    rows: ak.Array,
    file: str,
    field: str,
    group: str,
    tab_map: dict[int, str],
    with_units: bool = False,
) -> ak.Array:
    r"""Read the data from a particular field to an Awkward array.

    This replaces the TCM like object defined by the channels and rows with the
    corresponding data field.

    Parameters
    ----------
    channels
        Array of the channel indices (uids).
    rows
        Array of the rows in the files to gather data from.
    file
        File to read the data from.
    field
        the field to read.
    group
        the group to read data from (eg. `hit` or `stp`.)
    tab_map
        mapping between indices and table names. Of the form:

        .. code:: python

            {NAME: UID}

        For example:

        .. code:: python

            {"det001": 1, "det002": 2}

    Returns
    -------
    an array with the data, of the same same as the channels and rows.
    """
    # initialise the output
    data_flat = None
    tcm_rows_full = None

    # save the unflattening
    reorder = ak.num(rows)

    for tab_name, key in tab_map.items():
        # get the rows to read

        idx = ak.flatten(rows[channels == key]).to_numpy()
        arg_idx = np.argsort(idx)

        # get the rows in the flattened data we want to append to
        tcm_rows = np.where(ak.flatten(channels == key))[0]

        # read the data with sorted idx
        data_ch = lh5.read(f"{group}/{tab_name}/{field}", file, idx=idx[arg_idx])
        units = data_ch.attrs.get("units", None)
        data_ch = data_ch.view_as("ak")

        # sort back to order for tcm
        data_ch = data_ch[np.argsort(arg_idx)]

        # append to output
        data_flat = ak.concatenate((data_flat, data_ch)) if data_flat is not None else data_ch
        tcm_rows_full = (
            np.concatenate((tcm_rows_full, tcm_rows)) if tcm_rows_full is not None else tcm_rows
        )

    if len(data_flat) != len(tcm_rows_full):  # type: ignore[arg-type]
        msg = "every index in the tcm should have been read"
        raise ValueError(msg)

    # sort the final data
    data_flat = data_flat[np.argsort(tcm_rows_full)]  # type: ignore[arg-type,index]
    data_unflat = ak.unflatten(data_flat, reorder)

    if with_units and units is not None:
        return ak.with_parameter(data_unflat, "units", units)

    return data_unflat
