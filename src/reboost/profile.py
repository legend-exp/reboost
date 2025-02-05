from __future__ import annotations

import logging
import time

from dbetto import AttrsDict

log = logging.getLogger(__name__)


class ProfileDict(AttrsDict):
    """A class to store the results of time profiling."""

    def update_field(self, name: str, time_start: float) -> None:
        """Update the stored time.

        Parameters
        ----------
        name
            the name of the field to update. If it contains / this
            will be interpreted as subdictionaries.
        time_start
            the starting time of the block to evaluate
        """
        name_split = name.split("/")
        group = None
        dict_tmp = None

        time_end = time.time()

        for idx, name_tmp in enumerate(name_split):
            dict_tmp = self if (group is None) else dict_tmp[group]

            # if we are at the end and the name is not in the dictionary add it
            if (idx == len(name_split) - 1) and (name_tmp not in dict_tmp):
                dict_tmp[name_tmp] = time_end - time_start

            # append the time different
            elif (idx == len(name_split) - 1) and (name_tmp in dict_tmp):
                dict_tmp[name_tmp] = dict_tmp[name_tmp] + (time_end - time_start)

            # create a subdictionary
            elif name_tmp not in dict_tmp:
                dict_tmp[name_tmp] = {}

            group = name_tmp

    def __repr__(self):
        return f"ProfileDict({dict(self)})"

    def __str__(self):
        """Return a human-readable profiling summary."""
        return "\nReboost post processing took: \n" + self._format(self, indent=1)

    def _format(self, data: ProfileDict, indent: int = 1) -> str:
        """Recursively format the dictionary.

        Parameters
        ----------
        data
            The dictionary to format.
        indent
            The current indentation level.

        Returns
        -------
        the formatted print out.
        """
        output = ""
        space = " " * indent  # Indentation spaces

        for key, value in data.items():
            if isinstance(value, dict):  # If the value is a dictionary, recurse
                output += f"{space}- {key}:\n" + self._format(value, indent + 2)
            else:
                # Round floats to 1 decimal place
                value_print = round(value, 1) if isinstance(value, float) else value
                value_print = f"{value_print}".rjust(7) if value_print > 0 else "< 0.1".rjust(7)
                output += f"{space}- {key}".ljust(25)
                output += f": {value_print} s\n"

        return output
