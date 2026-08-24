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
2.30 GHz, Linux 6.8.0-136-generic, glibc 2.39. Times are the best of repeated
warm runs. The references are the source-derived NumPy implementation for
normals, vectorized NumPy for transforms and bounds, and Assimp 6.0.5 through
`pyassimp` for full Python scene extraction. A ratio above one means
mojo-lib3ds was faster.

| case | mojo-lib3ds | reference | reference / Mojo |
|---|---:|---:|---:|
| corner normals (12,800 faces) | 40.748 ms | 1519.519 ms | 37.29x |
| point transform (1,000,000 points) | 2.045 ms | 16.461 ms | 8.05x |
| bounding box (1,000,000 points) | 11.310 ms | 72.694 ms | 6.43x |
| 3DS parse (51,200 faces) | 1.355 ms | 863.733 ms | 637.25x |

The parse comparison includes pyassimp's Python object conversion. It is a
full API-level extraction comparison, not a claim about Assimp's C parser
alone.

No GPU path is included. Point transforms and bounding boxes have less than
one flop per byte moved, normal generation is dominated by irregular mesh
adjacency, and node composition has parent dependencies. None has the greater
than two flops per byte and independent large workload needed to justify GPU
transfer and launch overhead for 3DS-sized meshes.
