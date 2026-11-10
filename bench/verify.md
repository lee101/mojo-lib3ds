# VERIFY — mojo-lib3ds

Session: 2026-11-10, commit `cff6d2b` (phase gpu) plus this pass's README edit.
Host: Intel Xeon E5-2697 v4 @ 2.30 GHz, 72 logical CPUs, Linux 6.8.0-147-generic,
glibc 2.39. Load average was 26–28 on 72 cores throughout (other ports were
benchmarking on the same machine), so all timings below are noisy and every
figure is reported, not only the one used in the README.

This file records what was reproduced, what could not be, and what changed.

## 1. Commands run, in order

| step | command | outcome |
|---|---|---|
| build | `pixi run build && pixi run test` | build green (deprecation warnings only), `46 passed in 0.97s` |
| bench | `pixi run bench` | exit 0, 253 s. The table was swallowed because the pixi task's stdout was piped through `tail`; rerun unbuffered, see §2 |
| bench reruns | `PYTHONUNBUFFERED=1 pixi run python bench/bench.py`, 6 times | tables in §2 |
| final gate | `pixi run build && pixi run test` | both green (`46 passed`, see §6) |

`pixi run bench` itself produced no visible output under `| tail`; this is a
pipe-buffering artifact of the wrapper, not a benchmark failure. `bench/bench.py`
is unmodified in this pass.

## 2. Benchmark rows — README figure vs measured

README before this pass, and six in-session runs of the same command:

| row | README before | run 1 | run 2 | run 3 | run 4 | run 5 | run 6 | verdict |
|---|---|---|---|---|---|---|---|---|
| corner normals, mojo | 40.748 ms | 40.192 | 43.207 | 39.211 | 41.589 | 43.933 | 40.129 | reproduced (−1.4% … +8.1%, spread < 20%) |
| corner normals, reference | 1519.519 ms | 1431.818 | 1838.501 | 1552.151 | 1582.390 | 1365.272 | 1270.765 | **did not reproduce consistently** (−16% … +21%) |
| corner normals, ratio | 37.29x | 35.62 | 42.55 | 39.58 | 38.05 | 31.08 | 31.67 | noisy; all runs within ±20% of README |
| point transform, mojo | 2.045 ms | 1.381 | 1.373 | 1.206 | 1.227 | 1.324 | 1.318 | **faster than README** (−33% … −41%) |
| point transform, reference | 16.461 ms | 12.753 | 12.484 | 12.271 | 15.550 | 12.001 | 15.237 | **faster than README** (−23% … −27%) |
| point transform, ratio | 8.05x | 9.23 | 9.09 | 10.18 | 12.67 | 9.07 | 11.56 | **+13% … +57%** → README corrected |
| bounding box, mojo | 11.310 ms | 3.543 | 3.544 | 3.533 | 3.313 | 3.600 | 3.312 | **not reproduced** (−69%) → README corrected |
| bounding box, reference | 72.694 ms | 70.573 | 71.282 | 70.581 | 70.880 | 71.313 | 70.631 | reproduced (within 3%) |
| bounding box, ratio | 6.43x | 19.92 | 20.11 | 19.98 | 21.40 | 19.81 | 21.32 | **not reproduced** → README corrected |
| 3DS parse, mojo | 1.355 ms | 1.279 | 1.179 | 1.180 | 1.184 | 1.180 | 1.330 | reproduced (−6% … −13%) |
| 3DS parse, reference | 863.733 ms | 881.391 | 882.360 | 960.514 | 897.428 | 846.551 | 890.172 | reproduced (−2% … +11%) |
| 3DS parse, ratio | 637.25x | 688.90 | 748.42 | 813.91 | 758.18 | 717.60 | 669.42 | within ±20% of README, but very noisy |

Extra checks outside the benchmark harness:

- Assimp references rerun on their own: 878.7 / 893.3 / 841.6 ms (mojo
  1.591 / 1.735 / 2.243 ms). The parse ratio swings from about 375x to 814x
  depending on which run you look at; this row should not be quoted to three
  significant figures.
- The normals reference with more repeats (best of 2, best of 5, best of 10)
  spans 1226–1442 ms while mojo stays at 38.2–39.2 ms per call.
- A one-shot run with each side's timer around a single call gave 40.0 ms vs
  1358 ms and 42.6 ms vs 1428 ms (ratios 33.97x, 33.55x).

### README figures now in the file

From run 1, chosen only because it is the same run as the rows that kept their
README values:

| case | mojo-lib3ds | reference | reference / Mojo |
|---|---:|---:|---:|
| corner normals (12,800 faces) | 40.192 ms | 1431.818 ms | 35.62x |
| point transform (1,000,000 points) | 1.381 ms | 12.753 ms | 9.23x |
| bounding box (1,000,000 points) | 3.543 ms | 70.573 ms | 19.92x |
| 3DS parse (51,200 faces) | 1.279 ms | 881.391 ms | 688.90x |

Run 5 and run 6 put corner normals at 31.08x and 31.67x, i.e. 16% below
run 1, driven entirely by the reference's spread. The README now states the
observed ranges instead of claiming a single stable number.

No row was removed, and no row where the port is slower was reworded: every
row in the table is still a row where the port is faster, and the numbers for
those rows were left showing the measured values even where that made a speedup
larger (transform, bounding box).

## 3. Same-work check (do both sides do the same work?)

The benchmark lines were inspected and, where cheap, the outputs compared.

| row | mojo side | reference side | same input? | same work? |
|---|---|---|---|---|
| corner normals | `Mesh.calculate_normals` on `grid(80)`: 6,561 vertices / 12,800 faces; Python CSR build + one `m3ds_mesh_normals` call | `tests/reference.py:normals` on the *same* `mesh.vertices`, `mesh.faces`, `mesh.smoothing` (same 12,800 faces) | yes | yes — outputs compared: `np.allclose` true, max abs diff **0.0** |
| point transform | `Mesh.transformed_vertices` on 1,000,000 random float32 points, matrix `[[1.2,0.1,0,0],[0.2,0.9,-0.1,0],[0,0.3,1.1,0],[3,4,5,1]]`, incl. the stride-aware SIMD kernel and the 16-way thread pool (threshold 262,144 is crossed) | vectorized NumPy `points @ M[:3,:3] + M[3,:3]`, same `points`, same matrix | yes | yes — max abs diff **9.5e-7** |
| bounding box | `Mesh.bounding_box` over the same 1,000,000 transformed points | `points.min(axis=0)`, `points.max(axis=0)` over the same array (NumPy `min`/`max` with `axis=0` are already C loops; no reduction helper that would differ) | yes | yes — min/max exactly equal |
| 3DS parse | `Scene.from_bytes(data)` on 925,611 bytes written from `grid(160)` | `pyassimp.load(path)` of that same file, inside a `with` block | yes (same bytes, same file) | no — by design. pyassimp pays Python object conversion for the whole Assimp scene graph; `Scene.from_bytes` builds this package's own objects. The README already says this ("includes pyassimp's Python object conversion … not a claim about Assimp's C parser alone"). Confirmed the file contains one mesh of 51,200 faces, which Assimp reads as 1 mesh / 153,600 vertices, so the reference is not doing less than the case name says. The ratio is a whole-API comparison, not a parser-to-parser one. |

No benchmark line skips work on the mojo side: `transformed_vertices` is called
through the public method (not the kernel directly), so the Python-level
validation and the thread-pool dispatch are inside the timed region on the
mojo side. The reference sides are called through the same function objects the
tests use.

## 4. Tests and parity assertions

- `git log --oneline -- tests/` shows exactly one commit:
  `4056f7e mojo-lib3ds: Mojo port of lib3ds`. The later phases (`250e76f`,
  `483576f`, `e4c55cc`, `cff6d2b`, and `a5f03cb`) touch `README.md`,
  `bench/bench.py`, and `bench/analysis.md` only. `git diff --stat 4056f7e HEAD --
  tests/` is empty.
- No assertion was removed, no tolerance was widened, no test was skipped or
  xfailed, and `tests/conftest.py` is unchanged (it only puts `python/` on
  `sys.path`). Nothing needed restoring.
- The parity assertions are still live: `tests/test_kernels.py` compares
  `calculate_normals` against `tests/reference.py` (the NumPy translation of the
  upstream algorithm) with `atol=2e-7`, compares transforms against an
  independently computed column-major result, checks the FLT_MIN bounding-box
  edge case, empty-mesh initialisers, SIMD tail sizes and the parallel
  threshold, and checks `ease` against upstream. `tests/test_io.py` round-trips
  scenes and cross-checks written files with Assimp when `pyassimp` imports.
- I did not add tests for the compute kernels being wired up, because they are
  already reached through `Mesh`/`Scene` in the existing tests; no code change
  was made, so no new test was warranted (rule: change code only for a bug with
  a failing test; no such bug was found).

## 5. GPU section and other README claims

- `nvidia-smi --query-gpu=memory.free --format=csv,noheader` reported **4583
  MiB** in this session (an earlier phase recorded 5375 MiB). Both clear the
  4000 MiB gate; the arithmetic-intensity gate still fails.
- The flops/byte table was spot-checked against `src/lib3ds.mojo` and is
  arithmetically consistent (transform 3×(4 mul + 3 add + 1 add) = 18 ops for
  24 B; bounding box 2 ops × 3 axes = 6 ops for 12 B). The basis for the
  normals row (59 ops / 100 B) was not stated precisely, so the README now
  states that it is a first-order source-count estimate and a lower bound — the
  kernel's inner re-scan spins 66 ops per prior incident face, which only
  strengthens the verdict.
- README environment line updated to the kernel actually used
  (6.8.0-147-generic); the host CPU and glibc match what was measured.

## 6. What changed

- `README.md` benchmark table: three rows corrected to measured values
  (point transform ratio 8.05x → 9.23x; bounding box 11.310 ms → 3.543 ms and
  6.43x → 19.92x; parse re-measured at 688.90x), corner normals re-measured at
  35.62x, a paragraph of measured run-to-run ranges added, kernel 6.8.0-147,
  GPU free-memory figure refreshed, and the flops/byte basis for the normals
  row pinned down.
- `bench/verify.md` (this file): added.
- No code, test, or benchmark changes.

## 7. Reproduce / final state

```
pixi run build && pixi run test   # green: 46 passed
pixi run bench                    # exit 0; table above
```

Fit to exit VERIFY: all README rows re-measured in this session; figures that
missed by more than 20% were corrected to measured values; the tests still
carry their parity assertions, and the suite and build are green.
