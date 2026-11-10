"""Build one development variant and A/B it against the committed kernel.

Usage (from repo root):
  pixi run python bench/.phase/ab.py bench/.phase/var/v_simd.mojo

kernels=bbox   measure only the bounding box (default: both)
kernels=normals measure only the normals kernel
"""

from __future__ import annotations

import ctypes
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "bench" / ".phase" / "build"
I = ctypes.c_ssize_t

PROBES = '''

# ---- development probe wrappers, appended by bench/.phase/ab.py ----

@export("probe_bounding_box")
def probe_bounding_box(address: Int, point_count: Int, result_address: Int) abi("C"):
    mesh_bounding_box(address, point_count, result_address)


@export("probe_normals")
def probe_normals(
    vertices_address: Int,
    faces_address: Int,
    smoothing_address: Int,
    offsets_address: Int,
    adjacent_address: Int,
    face_normals_address: Int,
    result_address: Int,
    face_count: Int,
) abi("C"):
    mesh_normals(
        vertices_address,
        faces_address,
        smoothing_address,
        offsets_address,
        adjacent_address,
        face_normals_address,
        result_address,
        face_count,
    )
'''


def environment():
    env = os.environ.copy()
    if "MODULAR_HOME" not in env:
        mojo = subprocess.run(
            ["bash", "-lc", "command -v mojo"], capture_output=True, text=True
        ).stdout.strip()
        env["MODULAR_HOME"] = str(Path(mojo).parents[1] / "share" / "max")
    return env


def build(source: Path, tag: str):
    OUT.mkdir(parents=True, exist_ok=True)
    staged = OUT / f"{tag}.mojo"
    staged.write_text(source.read_text() + PROBES)
    target = OUT / f"lib{tag}.so"
    result = subprocess.run(
        ["mojo", "build", "--emit", "shared-lib", str(staged), "-o", str(target)],
        env=environment(),
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        lines = result.stderr.splitlines()
        keep = [
            line
            for line in lines
            if "error:" in line or "note: did you mean" in line
        ]
        raise SystemExit(f"build failed for {tag}:\n" + "\n".join(keep[:20]))
    library = ctypes.CDLL(str(target))
    library.probe_bounding_box.argtypes = [I, I, I]
    library.probe_bounding_box.restype = None
    library.probe_normals.argtypes = [I] * 8
    library.probe_normals.restype = None
    return library


def best(fn, repeat: int) -> float:
    value = float("inf")
    for _ in range(repeat):
        start = time.perf_counter()
        fn()
        value = min(value, time.perf_counter() - start)
    return value


def mesh_arrays(side: int):
    x, y = np.meshgrid(
        np.arange(side + 1, dtype=np.float32), np.arange(side + 1, dtype=np.float32)
    )
    z = np.sin(x * np.float32(0.03)) * np.cos(y * np.float32(0.03))
    vertices = np.column_stack((x.ravel(), y.ravel(), z.ravel())).astype(np.float32)
    cells = np.arange(side * side, dtype=np.int64).reshape(side, side)
    lower = cells + np.arange(side, dtype=np.int64)[:, None]
    a = lower.ravel()
    faces = np.column_stack(
        (
            np.concatenate((a, a)),
            np.concatenate((a + 1, a + side + 2)),
            np.concatenate((a + side + 2, a + side + 1)),
        )
    )
    smoothing = np.ones(len(faces), dtype=np.uint32)
    face_count = len(faces)
    counts = np.bincount(faces.ravel(), minlength=len(vertices))
    offsets = np.empty(len(vertices) + 1, dtype=np.int64)
    offsets[0] = 0
    np.cumsum(counts, out=offsets[1:])
    adjacent = np.empty(3 * face_count, dtype=np.int64)
    cursor = offsets[1:].copy()
    for face in range(face_count):
        for vertex in faces[face]:
            cursor[vertex] -= 1
            adjacent[cursor[vertex]] = face
    face_normals = np.empty((face_count, 3), dtype=np.float32)
    result = np.empty((face_count, 3, 3), dtype=np.float32)
    return vertices, faces, smoothing, offsets, adjacent, face_normals, result


def addr(array: np.ndarray) -> int:
    return int(array.ctypes.data)


def run_bounds(arms, which: str) -> tuple[dict, dict]:
    rng = np.random.default_rng(7)
    points = np.ascontiguousarray(rng.normal(size=(1_000_000, 3)), dtype=np.float32)
    bounds = np.empty(6, dtype=np.float32)
    expected_min = points.min(axis=0)
    expected_max = points.max(axis=0)
    print(f"bounding box, 1,000,000 points, best of 25 (ms) [{which}]")
    times = {}
    for tag, library in arms:
        fn = library.probe_bounding_box
        fn(addr(points), len(points), addr(bounds))
        ok = np.array_equal(bounds[:3], expected_min) and np.array_equal(
            bounds[3:], expected_max
        )
        times[tag] = best(lambda: fn(addr(points), len(points), addr(bounds)), 25)
        print(f"  {tag:20s} {times[tag] * 1e3:8.4f}  {'ok' if ok else 'MISMATCH'}")
    return times, {}


def run_normals(arms, which: str) -> tuple[dict, dict]:
    vertices, faces, smoothing, offsets, adjacent, face_normals, result = (
        mesh_arrays(80)
    )
    expected = None
    print("corner normals, grid(80) = 12,800 faces, best of 15 (ms)")
    times = {}
    for tag, library in arms:
        fn = library.probe_normals

        def run(fn=fn):
            fn(
                addr(vertices),
                addr(faces),
                addr(smoothing),
                addr(offsets),
                addr(adjacent),
                addr(face_normals),
                addr(result),
                len(faces),
            )

        run()
        if expected is None:
            expected = result.copy()
        difference = float(np.max(np.abs(result - expected)))
        times[tag] = best(run, 15)
        print(f"  {tag:20s} {times[tag] * 1e3:8.4f}  maxdiff-vs-src={difference:.2e}")
    return {}, times


def main(argv: list[str]) -> None:
    if not argv:
        raise SystemExit(__doc__)
    which = "both"
    specs = []
    for item in argv:
        if item.startswith("kernels="):
            which = item.split("=", 1)[1]
        else:
            specs.append(item)
    os.sched_setaffinity(0, {0})
    arms = [("src", build(ROOT / "src" / "lib3ds.mojo", "src"))]
    for spec in specs:
        path = Path(spec)
        arms.append((path.stem, build(ROOT / path, path.stem)))

    bound_times, normal_times = {}, {}
    if which in ("both", "bbox"):
        bound_times, _ = run_bounds(arms, which)
    if which in ("both", "normals"):
        _, normal_times = run_normals(arms, which)

    print("\nrelative to src (below 1.0 is faster)")
    for tag, _ in arms[1:]:
        parts = []
        if bound_times:
            parts.append(f"bounds {bound_times[tag] / bound_times['src']:6.3f}x")
        if normal_times:
            parts.append(f"normals {normal_times[tag] / normal_times['src']:6.3f}x")
        print(f"  {tag:20s} " + "   ".join(parts))
    print(f"loadavg: {os.getloadavg()}")


if __name__ == "__main__":
    main(sys.argv[1:])
