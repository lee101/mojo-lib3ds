"""Benchmarks against the independent Assimp reader and source-derived NumPy."""

from __future__ import annotations

import math
import os
import platform
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
from pyassimp import load as assimp_load

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))
sys.path.insert(0, str(ROOT / "tests"))

from mojo_lib3ds import Mesh, Scene  # noqa: E402
from reference import normals as reference_normals  # noqa: E402


def best(function, repeat: int = 3) -> float:
    result = math.inf
    for _ in range(repeat):
        start = time.perf_counter()
        function()
        result = min(result, time.perf_counter() - start)
    return result


def grid(side: int) -> Mesh:
    x, y = np.meshgrid(
        np.arange(side + 1, dtype=np.float32),
        np.arange(side + 1, dtype=np.float32),
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
    return Mesh("grid", vertices, faces, smoothing=smoothing)


def cpu_name() -> str:
    try:
        for line in Path("/proc/cpuinfo").read_text().splitlines():
            if line.startswith("model name"):
                return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor() or platform.machine()


def main() -> None:
    rows: list[tuple[str, float, float]] = []

    normal_mesh = grid(80)
    normal_mesh.calculate_normals()
    rows.append(
        (
            f"corner normals ({len(normal_mesh.faces):,} faces)",
            best(normal_mesh.calculate_normals),
            best(
                lambda: reference_normals(
                    normal_mesh.vertices, normal_mesh.faces, normal_mesh.smoothing
                ),
                repeat=2,
            ),
        )
    )

    rng = np.random.default_rng(7)
    points = np.ascontiguousarray(
        rng.normal(size=(1_000_000, 3)), dtype=np.float32
    )
    transform_mesh = Mesh("points", points, np.empty((0, 3), dtype=np.int64))
    matrix = np.array(
        [[1.2, 0.1, 0.0, 0], [0.2, 0.9, -0.1, 0], [0.0, 0.3, 1.1, 0], [3, 4, 5, 1]],
        dtype=np.float32,
    )
    transform_mesh.transformed_vertices(matrix)

    def numpy_transform():
        return (
            points @ np.ascontiguousarray(matrix[:3, :3])
            + matrix[3, :3]
        ).astype(np.float32)

    rows.append(
        (
            "point transform (1,000,000 points)",
            best(lambda: transform_mesh.transformed_vertices(matrix), repeat=5),
            best(numpy_transform, repeat=5),
        )
    )

    io_mesh = grid(160)
    data = Scene(meshes=[io_mesh]).to_bytes()
    with tempfile.NamedTemporaryFile(suffix=".3ds", delete=False) as temporary:
        temporary.write(data)
        path = temporary.name
    try:
        Scene.from_bytes(data)

        def assimp_parse():
            with assimp_load(path):
                pass

        assimp_parse()
        rows.append(
            (
                f"3DS parse ({len(io_mesh.faces):,} faces)",
                best(lambda: Scene.from_bytes(data), repeat=5),
                best(assimp_parse, repeat=5),
            )
        )
    finally:
        os.unlink(path)

    print(f"Machine: {cpu_name()}; {platform.platform()}")
    print()
    print("| case | mojo-lib3ds | reference | reference / Mojo |")
    print("|---|---:|---:|---:|")
    for name, mojo_time, reference_time in rows:
        ratio = reference_time / mojo_time
        print(
            f"| {name} | {mojo_time * 1e3:.3f} ms | "
            f"{reference_time * 1e3:.3f} ms | {ratio:.2f}x |"
        )


if __name__ == "__main__":
    main()
