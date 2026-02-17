from __future__ import annotations

import awkward as ak

from reboost.spms.pe import (
    corrected_photoelectrons,
    emitted_scintillation_photons,
    number_of_detected_photoelectrons,
    photoelectron_times,
)


def test_forced_trigger_correction():
    # check that with every data event empty it does nothing

    pe, uid = corrected_photoelectrons(
        ak.Array([[], [1], [2, 3]]), ak.Array([[], [0], [0, 1]]), ak.Array([[]]), ak.Array([[]])
    )

    assert ak.all(pe == [[], [1], [2, 3]])
    assert ak.all(uid == [[], [0], [0, 1]])

    # check adding a constant
    pe, uid = corrected_photoelectrons(
        ak.Array([[], [1], [2, 3]]), ak.Array([[], [0], [0, 1]]), ak.Array([[1]]), ak.Array([[0]])
    )

    assert ak.all(pe == [[1], [2], [3, 3]])
    assert ak.all(uid == [[0], [0], [0, 1]])

    # check sorting
    pe, uid = corrected_photoelectrons(
        ak.Array([[], [1], [2, 3]]),
        ak.Array([[], [0], [0, 1]]),
        ak.Array([[1, 3]]),
        ak.Array([[1, 0]]),
    )
    assert ak.all(pe == [[3, 1], [4, 1], [5, 4]])
    assert ak.all(uid == [[0, 1], [0, 1], [0, 1]])


def test_emitted_scintillation_photons():
    edep = ak.Array([[1.0, 2.0], [3.0]])
    particle = ak.Array([[22, 22], [22]])

    out = emitted_scintillation_photons(edep, particle, "lar")

    assert ak.num(out).tolist() == ak.num(edep).tolist()
    assert ak.all(out >= 0)
    assert ak.all(ak.values_astype(out, int) == out)


def test_number_of_detected_photoelectrons(mock_optmap_for_convolve):
    xloc = ak.Array([[0.1, 0.2], [0.3]])
    yloc = ak.Array([[0.1, 0.2], [0.3]])
    zloc = ak.Array([[0.1, 0.2], [0.3]])
    num_scint_ph = ak.Array([[10, 20], [30]])

    out, is_max = number_of_detected_photoelectrons(
        xloc,
        yloc,
        zloc,
        num_scint_ph,
        mock_optmap_for_convolve,
        "all",
    )

    assert is_max.tolist() == [False, False]
    assert ak.num(out).tolist() == ak.num(num_scint_ph).tolist()
    assert ak.all(out >= 0)
    assert ak.all(ak.values_astype(out, int) == out)


def test_number_of_detected_photoelectrons_max(mock_optmap_for_convolve):
    xloc = ak.Array([[0.1, 0.2], [0.3]])
    yloc = ak.Array([[0.1, 0.2], [0.3]])
    zloc = ak.Array([[0.1, 0.2], [0.3]])
    num_scint_ph = ak.Array([[1, 1], [3000]])

    out, is_max = number_of_detected_photoelectrons(
        xloc,
        yloc,
        zloc,
        num_scint_ph,
        mock_optmap_for_convolve,
        "all",
        photon_threshold_per_hit=5,
    )

    assert is_max.tolist() == [False, True]
    assert ak.num(out).tolist() == ak.num(num_scint_ph).tolist()
    assert ak.all(out >= 0)
    assert ak.all(ak.values_astype(out, int) == out)

    num_scint_ph = ak.Array([[1, 10000], [1]])

    out, is_max = number_of_detected_photoelectrons(
        xloc,
        yloc,
        zloc,
        num_scint_ph,
        mock_optmap_for_convolve,
        "all",
        photon_threshold_per_hit=5,
    )

    assert is_max.tolist() == [True, False]


def test_photoelectron_times():
    num_det_ph = ak.Array([[0, 2], [1]])
    particle = ak.Array([[22, 22], [22]])
    time = ak.Array([[0.0, 1.0], [2.0]])

    out = photoelectron_times(num_det_ph, particle, time, "lar")

    assert ak.num(out).tolist() == ak.sum(num_det_ph, axis=1).tolist()
    assert ak.all(out >= 0)
