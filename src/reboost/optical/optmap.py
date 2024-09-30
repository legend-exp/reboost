from __future__ import annotations

import logging
from collections.abc import Mapping

import hist
import numpy as np
from lgdo import Histogram, lh5

log = logging.getLogger(__name__)


class OpticalMap:
    def __init__(self, name: str, settings: Mapping[str, str]):
        self.settings = settings

        self.h_vertex = None
        self.h_hits = None
        self.h_prob = None
        self.h_prob_uncert = None
        self.name = name

    def create_empty(name: str, settings: Mapping[str, str]) -> OpticalMap:
        om = OpticalMap(name, settings)
        om.h_vertex = om._prepare_hist()
        om.h_hits = om._prepare_hist()
        return om

    @staticmethod
    def load_from_file(lh5_file: str, group: str) -> OpticalMap:
        om = OpticalMap(group, None)

        def read_hist(name: str, fn: str, group: str = "all"):
            h = lh5.read(f"/{group}/{name}", lh5_file=fn)
            if not isinstance(h, Histogram):
                msg = f"encountered invalid optical map while reading /{group}/{name} in {fn}"
                raise ValueError(msg)
            return h.view_as("hist")

        om.h_vertex = read_hist("nr_gen", lh5_file, group=group)
        om.h_hits = read_hist("nr_det", lh5_file, group=group)
        om.h_prob = read_hist("p_det", lh5_file, group=group)
        om.h_prob_uncert = read_hist("p_det_err", lh5_file, group=group)
        return om

    def _prepare_hist(self, **hist_kwargs) -> hist.Hist:
        """Prepare an empty histogram with the parameters global to this run."""
        bounds = self.settings["range_in_m"]
        bins = self.settings["bins"]
        return hist.Hist(
            hist.axis.Regular(
                bins=bins[0],
                start=bounds[0][0],
                stop=bounds[0][1],
                name="x",
                flow=False,
            ),
            hist.axis.Regular(
                bins=bins[1],
                start=bounds[1][0],
                stop=bounds[1][1],
                name="y",
                flow=False,
            ),
            hist.axis.Regular(
                bins=bins[2],
                start=bounds[2][0],
                stop=bounds[2][1],
                name="z",
                flow=False,
            ),
            **hist_kwargs,
        )

    def fill_vertex(self, loc) -> None:
        if self.h_vertex is None:
            self.h_vertex = self._prepare_hist()
        self.h_vertex.fill(loc[:, 0], loc[:, 1], loc[:, 2])

    def fill_hits(self, loc) -> None:
        if self.h_hits is None:
            self.h_hits = self._prepare_hist()
        self.h_hits.fill(loc[:, 0], loc[:, 1], loc[:, 2])

    def _divide_hist(self, h1: hist.Hist, h2: hist.Hist) -> tuple[hist.Hist, hist.Hist]:
        """Calculate the ratio (and its standard error) from two histograms."""
        h1 = h1.view()
        h2 = h2.view()

        ratio_hist = self._prepare_hist()
        ratio = ratio_hist.view()
        ratio_err_hist = self._prepare_hist()
        ratio_err = ratio_err_hist.view()

        ratio[:] = np.divide(h1, h2, where=(h2 != 0))
        ratio[h2 == 0] = -1  # -1 denotes no statistics.

        if np.any(ratio > 1):
            msg = "encountered cell(s) with more hits then primaries"
            raise RuntimeError(msg)

        # compute uncertainty according to Bernoulli statistics.
        # TODO: this does not make sense for ratio==1
        ratio_err[h2 != 0] = np.sqrt((ratio[h2 != 0]) * (1 - ratio[h2 != 0]) / h2[h2 != 0])
        ratio_err[h2 == 0] = -1  # -1 denotes no statistics.

        return ratio_hist, ratio_err_hist

    def create_probability(self) -> None:
        self.h_prob, self.h_prob_uncert = self._divide_hist(self.h_hits, self.h_vertex)

    def write_lh5(self, lh5_file: str, group: str = "all") -> None:
        def write_hist(h: hist.Hist, name: str, fn: str, group: str = "all"):
            lh5.write(
                Histogram(h, binedge_attrs={"units": "m"}),
                name,
                fn,
                group=group,
                wo_mode="write_safe",
            )

        write_hist(self.h_vertex, "nr_gen", lh5_file, group=group)
        write_hist(self.h_hits, "nr_det", lh5_file, group=group)
        write_hist(self.h_prob, "p_det", lh5_file, group=group)
        write_hist(self.h_prob_uncert, "p_det_err", lh5_file, group=group)

    def check_histograms(self, include_prefix: bool = False) -> None:
        log_prefix = "" if not include_prefix else self.name + " - "

        def _warn(fmt: str, *args):
            log.warning("%s" + fmt, log_prefix, *args)  # noqa: G003

        h_vertex = self.h_vertex.view()
        h_prob = self.h_prob.view()
        h_prob_uncert = self.h_prob_uncert.view()

        ncells = h_vertex.shape[0] * h_vertex.shape[1] * h_vertex.shape[2]

        missing_v = np.sum(h_vertex <= 0)  # bins without vertices.
        if missing_v > 0:
            _warn("%d missing_v %.2f %%", missing_v, missing_v / ncells * 100)

        missing_p = np.sum(h_prob <= 0)  # bins without hist.
        if missing_p > 0:
            _warn("%d missing_p %.2f %%", missing_p, missing_p / ncells * 100)

        non_phys = np.sum(h_prob > 1)  # non-physical events with probability > 1.
        if non_phys > 0:
            _warn(
                "%d voxels (%.2f %%) with non-physical probability (p>1)",
                non_phys,
                non_phys / ncells * 100,
            )

        # warnings on insufficient statistics.
        large_error = np.sum(h_prob_uncert > 0.01 * h_prob)
        if large_error > 0:
            _warn(
                "%d voxels (%.2f %%) with large relative statistical uncertainty (> 1 %%)",
                large_error,
                large_error / ncells * 100,
            )

        primaries_low_stats_th = 100
        low_stat_zero = np.sum((h_vertex < primaries_low_stats_th) & (h_prob == 0))
        if low_stat_zero > 0:
            _warn(
                "%d voxels (%.2f %%) with non reliable probability estimate (p=0 and primaries < %d)",
                low_stat_zero,
                low_stat_zero / ncells * 100,
                primaries_low_stats_th,
            )
        low_stat_one = np.sum((h_vertex < primaries_low_stats_th) & (h_prob == 1))
        if low_stat_one > 0:
            _warn(
                "%d voxels (%.2f %%) with non reliable probability estimate (p=1 and primaries < %d)",
                low_stat_one,
                low_stat_one / ncells * 100,
                primaries_low_stats_th,
            )
