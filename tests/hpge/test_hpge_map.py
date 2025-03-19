from __future__ import annotations

import json
from pathlib import Path

import legendhpges as hpges
import numpy as np
import pyg4ometry as pg4

from reboost.hpge.utils import ReadHPGeMap


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


def test_drift_time():
    bege, reg = create_test_registry()

    dt_file_obj = ReadHPGeMap(
        "test_files/B99000A_drift_time_map.lh5", "drift_times", reg.physicalVolumeDict["BEGe"]
    )

    assert dt_file_obj.xpos == 0.05
    assert dt_file_obj.ypos == 0
    assert dt_file_obj.zpos == -0.03
    assert (dt_file_obj.HPGeMap[0].r, dt_file_obj.HPGeMap[0].z, dt_file_obj.HPGeMap[0].val) == (
        0.0,
        0.0002,
        364.0,
    )
    assert round(dt_file_obj.get_map_value(0.0 + 0.05, 0, 0.0002 - 0.03), 4) == 364
    assert round(dt_file_obj.get_map_value(0.045, 0, -0.02), 5) == 218.0


if __name__ == "__main__":
    test_drift_time()
