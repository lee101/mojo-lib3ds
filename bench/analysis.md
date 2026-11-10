# ANALYZE — mojo-lib3ds

**Status: INCOMPLETE — no measurement run completed in this session.**
The single `pixi run bench` required by this phase did not execute. It sat blocked on the
shared bench lock for 6m21s and was then killed by `| tail` closing its stdout pipe. A later
20-poll watch (`flock -n` probe every 15s, 5 minutes) never saw the lock free, so the lock is
held continuously by processes from other ports on this machine.

No row below is a measurement. No number is estimated. The two decision tables in §2 and §4
are derived from source code only and are explicitly labelled as such.

---

## 1. What was run

| step | command | outcome |
|---|---|---|
| 1 | `pixi run bench` (first attempt, `\| tail -20`) | queued on `/tmp/mojo-bench.lock`. No output reached the terminal before the pipe closed; task exited 0 after 381s without running the benchmark body |
| 2 | `flock -n /tmp/mojo-bench.lock true` | exit 1 (held). `fcntl.flock(LOCK_EX \| LOCK_NB)` → `[Errno 11] Resource temporarily unavailable` |
| 3 | 20 × (`timeout 5 flock -n … true`; `sleep 15`) | never free in 5 minutes; all 20 polls failed |
| 4 | `/proc/locks` inspection | see §1.1 |

Consequences: the `pixi run bench` results, the upstream/plain-NumPy comparisons at
"real caller" sizes, and the per-entry-point timings could not be produced. Everything in
§2–§4 is code reading for the next phases to confirm or discard against real runs.

### 1.1 Lock contention evidence

- Lock file is shared across ports: `/tmp/mojo-bench.lock`, inode 124763, empty, one hard link.
- `/proc/locks` shows one flock queue on that inode at a time, and the queued commands have
  working directories under other ports, not this repo — e.g.
  `/nvme0n1-disk/code/mojo-libigl`, `mojo-meshlab`, `mojo-h2`, `mojo-geohash2`,
  `mojo-mercantile`, `mojo-librosa`, `mojo-karateclub`, `mojo-libwebp`. 39 bench-like
  processes were live system-wide during the watch window.
- Machine load average during the session was 21–28 on 72 cores, rising from other ports'
  benchmarks and editors. Any benchmark taken in this window would not have been
  representative even if the lock had been free.

Note: by-inode filtering of `/proc/locks` was unreliable in this environment — a lock
held by a process that demonstrably blocked `flock -n` did not appear under that method.
The authoritative check used here is the acquire attempt itself (`flock -n`, exit code).

---

## 2. Entry points in scope (inventory from source, not benchmarked)

Source of public surface: `python/mojo_lib3ds/model.py` and `bench/bench.py`.
Sizes in the table are the main bench's in-session construction sizes (mesh geometry
is deterministic from `grid(side)` in `bench/bench.py`); they are inputs this phase
specifies, not timings.

| entry point | path | input size used by bench | kernel exported |
|---|---|---|---|
| `Mesh.calculate_normals` | Python CSR build + 1 ctypes call | 12,800 faces / 6,561 verts (`grid(80)`) | `m3ds_mesh_normals` |
| `Mesh.transformed_vertices` | 16-way `ThreadPoolExecutor`, 1 ctypes call per chunk | 1,000,000 points | `m3ds_transform_points` |
| `Mesh.bounding_box` | 1 ctypes call | 1,000,000 points | `m3ds_mesh_bounding_box` |
| `Scene.to_bytes` | pure Python (`io.py`) | 51,200 faces (`grid(160)`) | none |
| `Scene.from_bytes` | pure Python (`io.py`) | 51,200 faces | none |
| `Mesh.__post_init__` | numpy validation / contiguity / range checks | per construction; runs again on every `Scene.from_bytes` mesh | none |
| `Track.evaluate` | pure Python + small numpy | per key count (small) | `m3ds_ease` declared, unused here |
| `Scene.evaluate` | pure Python, O(n^2) topo-sort loop | node count | `m3ds_compose_nodes` |

---

## 3. Hotspot table (blocked — fields for a completed run)

The required columns, kept so the next run fills them rather than redesigning the table:

| function | input size | time | bytes touched | flops | flops/byte | bound | glue vs kernel | marked vs upstream |
|---|---|---|---|---|---|---|---|---|
| (unmeasured — bench never acquired the lock) | | | | | | | | |

No row may be filled from the README table: those numbers come from a different machine
(E5-2697 v4, kernel 6.8.0-136) and a different session, and the phase rules forbid reuse.

Optional upstream comparison: `import lib3ds` was not tested because the run needed to
establish it never started. Treat upstream availability as unknown until step 1 succeeds.

---

## 4. Source-level findings (no timings — verify before acting)

### 4.1 Python-level loops and per-call work

| site | finding | cost shape | evidence |
|---|---|---|---|
| `Mesh.calculate_normals`, `model.py:150-153` | pure Python double loop over every face and its 3 indices, one numpy scalar store per element (`adjacent[cursor[vertex]] = face`) | O(3F) Python iterations + 3F scalar numpy indexing round-trips; 38,400 element-stores at 12,800 faces | source |
| same | per-call allocations: `bincount`, `offsets`, `adjacent`, `cursor` copy, `face_normals`, `result` — the CSR build allocates for the *reference's* input shape, not the kernel's | 6 allocations per call; `adjacent`/`offsets` are rebuilt on every call and never cached on the `Mesh` | source |
| `Mesh.transformed_vertices`, `model.py:189-203` | builds `chunk_size` and a futures list, submits 16 chunks, then `future.result()` in order; `address(self.vertices[start:])` creates a view per chunk | 16 Python/thread-pool round-trips + 16 ctypes calls per call above 262,144 points (1M points → 16 chunks) | source |
| same | allocates the full result array each call (`np.empty_like`) | 12 MB for 1M points, unavoidable for the current API | source |
| `Mesh.bounding_box` | one ctypes call, but `.copy()` ×2 on the 3-element slices and an `np.empty(6)` | negligible per call; **call count matters if called per mesh** | source |
| `Scene.evaluate` (`model.py:389+`) | topological order via `remaining` set and an inner `for index in list(remaining)` that restarts until no progress | O(N^2) in node count, pure Python, per call | source |
| `Track.evaluate` (`model.py:287-319`) | rebuilds `incoming`/`outgoing` arrays and re-derives every tangent on **each** call even though only the two bracketing keys are needed; `sorted(self.keys …)` per call | O(K) allocations per evaluation, K = key count | source |
| `Scene.from_bytes` → `Mesh.__post_init__` | `_f32` runs `np.isfinite(source).all()` and `np.any(np.abs(source.astype(np.float64)))` — a float64 upcast of the whole vertex array on every mesh construction, including parse | extra full-size pass + 2× memory traffic per parse; pure Python/numpy, not the parser | source |

Direct answer to "loops that re-enter ctypes per element": **none.** Every kernel crossing is
one ctypes call per whole array (or per chunk). The Python-loop cost is *numpy element
indexing* inside `calculate_normals`, which is Python-level but not ctypes. The ctypes
re-entry pattern that does exist is per-**chunk** in `transformed_vertices`.

### 4.2 Arithmetic intensity (lower bound from geometry; needs measured bytes for the table)

| kernel | work per element | bytes streamed (min) | flops/byte | classification |
|---|---|---|---|---|
| `m3ds_transform_points` | 1 point: 9 fma + 3 add (+ 4th column) ≈ 12–15 flops per 3 floats | in 12 B + out 12 B = 24 B | ≈ 0.5–0.6 | **memory-bound** |
| `m3ds_mesh_bounding_box` | 1 mult + 3 compares per point, 3 floats | in 12 B → 0 out stream | ≈ 0.3 | **memory-bound** (read-only stream) |
| `m3ds_mesh_normals` | per face: edge cross + normalize per corner, plus neighbour smoothing over incident faces | vertices + faces + CSR indices + 3F×3 out | low, irregular | **memory/irregular-adjacency-bound**; not compute-bound |
| `m3ds_compose_nodes` | 4×4 matrix multiply per node | small | irrelevant at N ≤ hundreds | **latency/glue-bound** |

Every 3DS-sized workload here is below 2 flops/byte, matching the README's own
reasoning. That is the figure a GPU verdict hinges on, and it is a structural argument,
not a measured one — the measured bytes column is still empty.

---

## 5. Next-phase routing (proposal, pending measurement)

| site | proposed phase | rationale |
|---|---|---|
| `calculate_normals` CSR build (Python loop) | **unroll** — vectorise the CSR build (sort/repeat/argsort in numpy, or move to Mojo) | removes O(3F) Python iterations without touching the kernel or its results order |
| `m3ds_space_normals` inner kernel | **none** until measured; then simd if the per-face normalize shows scalar reduction | already ~37× over the NumPy reference on the README machine; need the current run to see glue vs kernel split |
| `transformed_vertices` | **parallel** — reconsider worker count / chunking using a measured threshold; currently 16 threads on 72 cores for 1M points | 16 is a constant, not a measured optimum; only worth changing after timing glue vs kernel |
| `transformed_vertices` kernel | **none** (already SIMD, stride-aware) | memory-bound at ≈0.5 flops/byte |
| `bounding_box` | **none** (kernel); **parallel** only if callers call it per mesh in a loop | read-only stream; GPU not justified at < 1 flops/byte |
| parse / `to_bytes` | **none** in this port's compute phases | branch-heavy Python by design; measure before any move |
| `Track.evaluate`, `Scene.evaluate` | **none** / **unroll** respectively, only if a real caller shows them hot | small N; O(N^2) set loop is the only structural concern |
| any GPU path | **gpu: not justified** on current evidence | no kernel exceeds 2 flops/byte by the structural count; GPU verdict must be re-taken once the measured table exists |

### 5.1 GPU phase verdict (taken this session)

The GPU gate was evaluated and closed as **CPU only: memory-bound**; see the
GPU section of the README. Recounted from the kernel bodies in
`src/lib3ds.mojo` (not from the README table): `m3ds_transform_points` 0.75,
`m3ds_mesh_bounding_box` 0.50, `m3ds_mesh_normals` 0.59, `m3ds_compose_nodes`
1.00 flops/byte — every kernel is below the 2 flops/byte gate, so no device
work was attempted. `nvidia-smi` reported 5375 MiB free, so the shared-GPU
memory gate itself passed; it was the arithmetic gate that failed.

No GPU timing was taken, and none is claimed. This does not depend on the
blocked bench: the gate is arithmetic intensity, counted statically. The
`pixi run bench` precondition was still unavailable this session —
`flock -n /tmp/mojo-bench.lock` returned exit 1 with load average 22–28 — so
the §3 measured table and §5 routing proposals remain unmeasured as before.

---

## 6. What unblocks this phase

1. Run `pixi run bench` when `/tmp/mojo-bench.lock` is free and load is sane
   (`flock -n /tmp/mojo-bench.lock true` → exit 0), producing its Markdown table.
2. Add per-entry-point timing at caller sizes (e.g. `grid(80)` and `grid(160)` normals,
   parse of both, 1k and 1M point transform/bounds) — a new bench script, since the
   existing bench reports one size per case.
3. Bracket each ctypes call with `perf_counter` to split glue from kernel, and count calls
   per public entry point (16 for a 1M-point transform today).
4. Split the flops/byte table into measured rows using kernel sizes from step 2.

Only file added by this phase: `bench/analysis.md`. Nothing in `src/` or `tests/` was
modified.
