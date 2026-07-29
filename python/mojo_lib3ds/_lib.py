from __future__ import annotations

import ctypes
import subprocess
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
LIBRARY = ROOT / "dist" / "libmojo-lib3ds.so"
I = ctypes.c_ssize_t
F = ctypes.c_float


def _load() -> ctypes.CDLL:
    if not LIBRARY.exists():
        subprocess.run(["bash", str(ROOT / "build" / "build.sh")], check=True)
    library = ctypes.CDLL(str(LIBRARY))
    library.m3ds_mesh_normals.argtypes = [I] * 8
    library.m3ds_mesh_normals.restype = None
    library.m3ds_mesh_bounding_box.argtypes = [I, I, I]
    library.m3ds_mesh_bounding_box.restype = None
    library.m3ds_transform_points.argtypes = [I, I, I, I]
    library.m3ds_transform_points.restype = None
    library.m3ds_compose_nodes.argtypes = [I] * 7
    library.m3ds_compose_nodes.restype = I
    library.m3ds_ease.argtypes = [F] * 5
    library.m3ds_ease.restype = F
    return library


lib = _load()


def address(array: np.ndarray) -> int:
    return int(array.ctypes.data)
