from __future__ import annotations

import logging
from typing import Any

import awkward as ak
from lgdo.types import LGDO, Table

from . import utils

log = logging.getLogger(__name__)


def evaluate_expression(
    hit_table: Table,
    expression: str,
    local_dict: dict,
    *,
    table_name: str = "STEPS",
) -> LGDO:
    """Evaluate an expression returning an LGDO.

    Uses :func:`LGDO.Table.eval()` to compute a new column for the
    hit table. The expression can depend on any field in the Table
    (prefixed with table_name.) or objects contained in the local dict.
    In addition, the expression can use packages which are then imported.

    Parameters
    ----------
    hit_table
        the table containing the hit fields.
    expression
        the expression to evaluate.
    local_dict
        local dictionary to pass to :func:`LGDO.Table.eval()`.
    table_name
        keyword used to refer to the fields in the table.

    Returns
    -------
        an LGDO with the new field.
    """

    raise NotImplementedError


def evaluate_object(
    expression: str,
    local_dict: dict,
) -> Any:
    """Evaluate an expression returning any object.

    The expression can depend on any objects contained in the local dict.
    In addition, the expression can use packages which are then imported.

    Parameters
    ----------
    expression
        the expression to evaluate.
    local_dict
        local dictionary to pass to :func:`LGDO.Table.eval()`.

    Returns
    -------
        the evaluated object.
    """

    raise NotImplementedError


def get_global_objects(expressions: dict[str, str], *, local_dict: dict) -> dict:
    """Extract global objects used in the processing.

    Parameters
    ----------
    expressions
        a dictionary containing the expressions to evaluate for each object.
    local_dict
        other objects used in the evaluation of the expressions, passed to
        `eval()` as the locals keyword.

    Returns
    -------
    dictionary of objects with the same keys as the expressions.
    """
    return {
        obj_name: evaluate_object(expression, local_dict=local_dict)
        for obj_name, expression in expressions.items()
    }


def get_detectors_mapping(
    output_detector_expression: str, input_detector_map_expression: str | None, objects: dict
) -> dict:
    """Extract the output detectors and the list of input to outputs."""
    raise NotImplementedError


def get_detector_objects(output_detectors, proc_group, args, global_objects):
    """Get the detector objects."""
    det_objects_dict = {}
    for output_detector in output_detectors:
        det_objects_dict[output_detector] = utils.dict2tuple(
            {
                obj_name: evaluate_object(
                    obj_expression,
                    local_dict={
                        "ARGS": args,
                        "DETECTOR": output_detector,
                        "OBJECTS": global_objects,
                    },
                )
                for obj_name, obj_expression in proc_group.get("detector_objects", {}).items()
            }
        )
    return utils.dict2tuple(det_objects_dict)


def evaluate_hit_table_layout(steps_ak: ak.Array, expression: str) -> Table:
    """Evaluate the hit_table_layout expression, producing the hit table.

    This expression should perform a restructuring of the steps, i.e. it sets
    the number of rows.

    Parameters
    ----------
    steps_ak
        awkward array of the steps.
    expression
        the expression to evaluate to produce the hit table.

    Returns
    -------
    Table of the hits.
    """
    raise NotImplementedError


def remove_columns(tab: Table, outputs: list) -> Table:
    """Remove columns from the table not found in the outputs.

    Parameters
    ----------
    tab
        the table to remove columns from.
    outputs
        a list of output fields.

    Returns
    -------
    the table with columns removed.
    """

    existing_cols = list(tab.keys())
    for col in existing_cols:
        if col not in outputs:
            tab.remove_column(col, delete=True)
    return tab
