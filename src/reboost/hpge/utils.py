from __future__ import annotations

from math import floor

import numpy as np
from lgdo import lh5
from pyg4ometry.geant4 import LogicalVolume


class HPGeMapPoint:
    def __init__(self, r: np.float64, z: np.float64, val: np.float64) -> None:
        self.r = r
        self.z = z
        self.val = val


class ReadHPGeMap:
    def __init__(self, filename: str, field_name: str, hpge: LogicalVolume) -> None:
        det_pos_x, det_pos_y, det_pos_z = [float(val) for val in hpge.position]
        self.xpos = det_pos_x
        self.ypos = det_pos_y
        self.zpos = det_pos_z

        self.filename = filename

        dt_data, _ = lh5.LH5Store().read(f"/{field_name}", filename)
        unit_conversion = {"um": 1e-6, "mm": 1e-3, "cm": 1e-2, "m": 1}

        self.r = dt_data.r.nda * unit_conversion[dt_data.r.attrs["units"]]  # force it to be in m
        self.z = dt_data.z.nda * unit_conversion[dt_data.z.attrs["units"]]  # force it to be in m
        self.val = dt_data.val.nda

        self.HPGeMap = [HPGeMapPoint(r, z, val) for r, z, val in zip(self.r, self.z, self.val)]

        self.compute_grid()

    def compute_grid(self) -> None:
        """Computes grid parameters for interpolation."""
        rs, zs = np.unique(self.r), np.unique(self.z)
        self.dz = abs(zs[1] - zs[0])
        self.dr = abs(rs[1] - rs[0])
        self.maxz = max(zs)
        self.maxr = max(rs)
        self.minz = min(zs)
        self.minr = min(rs)

    def get_map_index(self, r: np.float64, z: np.float64) -> int:
        rGrid = floor(r / self.dr + 0.5) * self.dr
        zGrid = floor(z / self.dz + 0.5) * self.dz
        numZ = int((self.maxz - self.minz) / self.dz + 1)
        i = int((rGrid - self.minr) / self.dr * numZ + (zGrid - self.minz) / self.dz)
        nDT = len(self.HPGeMap)

        if i >= nDT:
            return nDT - 1
        return i

    def get_map_value(self, x: np.float64, y: np.float64, input_z: np.float64) -> np.float64:
        r = self.get_r(x, y)
        z = input_z - self.zpos

        r1 = floor(r / self.dr) * self.dr
        z1 = floor(z / self.dz) * self.dz

        val_11 = self.HPGeMap[self.get_map_index(r1, z1)].val
        if r1 == r and z1 == z:
            return val_11

        # Bilinear interpolation. Might do cubic interpolation eventually.
        r2 = r1 + self.dr
        z2 = z1 + self.dz

        val_12 = self.HPGeMap[self.get_map_index(r1, z2)].val
        val_21 = self.HPGeMap[self.get_map_index(r2, z1)].val
        val_22 = self.HPGeMap[self.get_map_index(r2, z2)].val

        return (
            1
            / ((r2 - r1) * (z2 - z1))
            * (
                val_11 * (r2 - r) * (z2 - z)
                + val_21 * (r - r1) * (z2 - z)
                + val_12 * (r2 - r) * (z - z1)
                + val_22 * (r - r1) * (z - z1)
            )
        )

    def get_r(self, x: np.float64, y: np.float64) -> np.float64:
        return np.sqrt((x - self.xpos) ** 2 + (y - self.ypos) ** 2)
