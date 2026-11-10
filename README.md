# mojo-lib3ds

`mojo-lib3ds` is a standalone Mojo/Python implementation of the useful core of
[lib3ds 1.3.0](https://lib3ds.sourceforge.net/), the classic Autodesk 3DS
chunk-file toolkit. It reads and writes `.3ds` files, exposes meshes,
materials, cameras, lights, and animation nodes as Python objects, and moves
compute-heavy mesh and hierarchy work into a compiled Mojo shared library.

This is an independent implementation informed by lib3ds behavior and file
format handling; it does not contain or link lib3ds code. The repository is
MIT licensed. See [LICENSE](LICENSE) and [NOTICE](NOTICE). The format is
associated with Autodesk products, but neither lib3ds nor this project is
affiliated with Autodesk.

## Install

The checked-in Pixi environment pins the Mojo nightly used to build the
library:

```sh
pixi install
pixi run build
pixi run test
```

## Usage

```python
import numpy as np
from mojo_lib3ds import Material, Mesh, Scene

mesh = Mesh(
    "triangle",
    np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=np.float32),
    np.array([[0, 1, 2]], dtype=np.int64),
    smoothing=np.array([1], dtype=np.uint32),
    face_materials=["red"],
)
scene = Scene(meshes=[mesh], materials=[Material(name="red")])
scene.save("triangle.3ds")

loaded = Scene.load("triangle.3ds")
print(loaded.meshes[0].calculate_normals())
```

Save this as `example.py` and run it with `pixi run python example.py` after
building.

## Tested coverage

The test suite proves read/write round trips for:

- triangle-mesh points, faces, smoothing groups, per-face materials, UV
  coordinates, and object flags;
- ambient, diffuse, and specular material colors, shininess, transparency,
  two-sided state, and the first texture map;
- camera position, target, roll, field of view, ranges, and cone state;
- spotlight position, target, color, multiplier, cone angles, and shadow state;
- object-node IDs and parents, position/rotation/scale tracks, hierarchy
  evaluation, non-cyclic vector TCB interpolation, and cumulative axis-angle
  rotation keys;
- empty scenes and meshes, malformed chunk rejection, mesh size limits, and
  interoperability of emitted mesh files with Assimp.

The compute tests compare smoothing-group corner normals and ease evaluation
with source-derived NumPy references. They also cover degenerate and
high-valence geometry, bounding boxes, column-major point transforms across
SIMD tail sizes and the parallel threshold, and object-node matrix composition.

Compared with upstream lib3ds, this project does not cover
background/atmosphere/shadow or viewport records, mesh box maps and procedural
mapping records, the less common material map slots, morph/hide/color/scalar
track APIs, cyclic smoothing endpoint setup, exact quaternion TCB `squad`
interpolation, or upstream's allocation and low-level I/O APIs. Rotation
evaluation uses normalized quaternion interpolation between cumulative
rotations. Unknown chunks are validated and skipped but are not retained when
a scene is written again.

The 3DS format limits each mesh to 65,535 points and 65,535 faces. In-memory
`Mesh` objects may be larger for compute kernels, but `Scene.to_bytes()`
rejects a mesh that cannot be represented.

## Correctness

There is no maintained installable Python binding to this lib3ds release.
Tests therefore use two references:

- Assimp through `pyassimp` independently reads files emitted by the writer;
- a NumPy translation follows the upstream normals, normalization, and ease
  algorithms.

All arrays crossing the native boundary are validated for shape, numeric range,
dtype, and contiguity. NumPy retains ownership for the complete synchronous
call; the shared library neither allocates nor retains Python buffers.

## How it works

Parsing and serialization stay in Python because they are branch-heavy and
mostly allocate Python scene objects. Chunk boundaries are validated before
payload access.

Hot operations cross a small C ABI into one Mojo compilation unit. Vertices,
matrices, and normals are packed `float32`; indices and hierarchy links are
`int64`; smoothing masks are `uint32`. Mesh adjacency uses a flat CSR index.
Matrices retain lib3ds's `[column, row]` layout. Point transforms use
stride-aware SIMD directly on interleaved buffers and split large inputs
across a reusable host thread pool.

## Benchmarks

Measured by `pixi run bench` on this machine: Intel Xeon E5-2697 v4 at
2.30 GHz, Linux 6.8.0-147-generic, glibc 2.39. Times are the best of repeated
warm runs. The references are the source-derived NumPy implementation for
normals, vectorized NumPy for transforms and bounds, and Assimp 6.0.5 through
`pyassimp` for full Python scene extraction. A ratio above one means
mojo-lib3ds was faster.

| case | mojo-lib3ds | reference | reference / Mojo |
|---|---:|---:|---:|
| corner normals (12,800 faces) | 40.192 ms | 1431.818 ms | 35.62x |
| point transform (1,000,000 points) | 1.381 ms | 12.753 ms | 9.23x |
| bounding box (1,000,000 points) | 3.543 ms | 70.573 ms | 19.92x |
| 3DS parse (51,200 faces) | 1.279 ms | 881.391 ms | 688.90x |

These are the figures from one run of the unchanged `bench/bench.py` in the
verify pass. Repeated runs of the same command move them: mojo-lib3ds normals
40.1–43.9 ms, transform 1.21–1.38 ms, bounding box 3.31–3.60 ms, parse
1.18–1.33 ms; the references move by up to about ±20% (corner normals
1271–1838 ms, Assimp parse 847–961 ms). The machine was shared and loaded, so
treat ratios as approximate to that extent. `bench/verify.md` records every
run.

The parse comparison includes pyassimp's Python object conversion. It is a
full API-level extraction comparison, not a claim about Assimp's C parser
alone.

## GPU: CPU only: memory-bound

No GPU path is included. The GPU phase gate requires a kernel above roughly
2 flops per byte before any device work is attempted. Every kernel in
`src/lib3ds.mojo` is below it, counted from the kernel bodies (mul and add
counted individually, `float32` at 4 B, `int64` at 8 B):

| kernel | flops (per element) | bytes moved | flops/byte | bound |
|---|---:|---:|---:|---|
| `m3ds_transform_points` | 18 | 24 (12 in + 12 out) | 0.75 | memory |
| `m3ds_mesh_bounding_box` | 6 | 12 (read-only stream) | 0.50 | memory |
| `m3ds_mesh_normals` | 59 | 100 | 0.59 | memory / irregular adjacency |
| `m3ds_compose_nodes` | 128 | 128 | 1.00 | latency, small N |

`m3ds_mesh_normals` also re-reads `face_normals` and `adjacent[]` per corner
through CSR offset ranges, so its effective byte count on real meshes is
higher than the 100 shown and its intensity is lower. Counting only the spins
inside its inner re-scan loop gives 66 ops per prior incident face, so the
ratio in the table is a lower bound. The 59/100 basis is the kernel's
first-order work per corner (3 corners × (8 mul + 5 add) for the face normal
and accumulation, plus 12 B in and 12 B out per corner); it is a source-count
estimate, not a profiled figure. `m3ds_compose_nodes` runs
on hundreds of nodes at most and has parent dependencies, so it has no
independent large workload to send to a device.

Gates checked in this session:

- `nvidia-smi --query-gpu=memory.free --format=csv,noheader` reported
  **4583 MiB** free in the verify session (an earlier phase recorded
  5375 MiB). Both are above 4000 MiB, so the device-availability gate passes;
  the arithmetic-intensity gate did not.
- Per the gate rule, the verdict is recorded here and no GPU code was written.
  No device allocation, kernel launch, or GPU measurement exists for this port,
  and no number in this section is a timing.

The CPU path remains the only path. No GPU test is emitted, so the suite has
nothing to skip on this or any other host: `pixi run test` reports 46 passed,
which is the complete CPU suite, with zero skips and zero GPU tests.
