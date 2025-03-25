from __future__ import annotations

import json
from pathlib import Path

import legendhpges as hpges
import numpy as np
import pyg4ometry as pg4
from lgdo import Table, lh5

from reboost.hpge.psd import do_cluster, dt_heuristic
from reboost.hpge.utils import ReadHPGeMap
from reboost.shape.group import group_by_time


def create_test_registry():
    reg = pg4.geant4.Registry()
    json_path = Path("test_files/B99000A.json")
    with json_path.open("r") as file:
        bege_meta = json.load(file)

    # create logical volumes for the two HPGe detectors
    bege_l = hpges.make_hpge(bege_meta, name="BEGe_L", registry=reg)

    # create a world volume
    world_s = pg4.geant4.solid.Orb("World_s", 20, registry=reg, lunit="cm")
    world_l = pg4.geant4.LogicalVolume(world_s, "G4_Galactic", "World", registry=reg)
    reg.setWorld(world_l)

    # let's make a liquid argon balloon
    lar_s = pg4.geant4.solid.Orb("LAr_s", 15, registry=reg, lunit="cm")
    lar_l = pg4.geant4.LogicalVolume(lar_s, "G4_lAr", "LAr_l", registry=reg)
    pg4.geant4.PhysicalVolume([0, 0, 0], [0, 0, 0], lar_l, "LAr", world_l, registry=reg)

    # now place the two HPGe detectors in the argon
    pg4.geant4.PhysicalVolume([0, 0, 0], [0.05, 0, -0.03, "m"], bege_l, "BEGe", lar_l, registry=reg)

    # finally create a small radioactive source
    source_s = pg4.geant4.solid.Tubs("Source_s", 0, 1, 1, 0, 2 * np.pi, registry=reg)
    source_l = pg4.geant4.LogicalVolume(source_s, "G4_BRAIN_ICRP", "Source_L", registry=reg)
    pg4.geant4.PhysicalVolume([0, 0, 0], [0, 0, 0, "m"], source_l, "Source", lar_l, registry=reg)
    return bege_l, reg


def test_dt_heuristic():
    data = lh5.read_as("stp/det001/", "test_files/internal_electron.lh5", "ak")
    grouped_data = group_by_time(data, evtid_name="evtid").view_as("ak")
    bege, reg = create_test_registry()

    dt_file_obj = ReadHPGeMap(
        "test_files/B99000A_drift_time_map.lh5", "drift_times", reg.physicalVolumeDict["BEGe"]
    )
    cluster_size_mm = 0.1

    _non_clustered_dth = dt_heuristic(grouped_data, dt_file_obj)
    cluster_data = do_cluster(grouped_data, cluster_size_mm, dt_file_obj)
    _clustered_dth = dt_heuristic(cluster_data, dt_file_obj)

    sto = lh5.LH5Store()

    sto.write(
        Table({"dth": _non_clustered_dth}),
        "/non_clustered_dth",
        "test_files/internal_electron_dt.lh5",
    )
    sto.write(
        Table({"dth": _clustered_dth}),
        "/clustered_dth",
        "test_files/internal_electron__clust_dt.lh5",
    )


if __name__ == "__main__":
    test_dt_heuristic()
