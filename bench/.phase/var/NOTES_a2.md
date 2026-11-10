"""CANDIDATE a2: 48 floats (16 points) per iteration, group-of-3 lane folding.

Layout facts (W = simd_width_of[float32]()):
  point p has (x, y, z) at flat offsets 3p, 3p+1, 3p+2.
  A flat offset f lives in vector (f // W), lane (f % W).
  So within the accumulator, lane L holds axis (f % 3) of point f // 3 for the
  point whose flat index f satisfied f % W == L.
  Folding: lo = min(lo, lo.shuffle[shift...]) over all shifts folds every lane
  into every lane; taking the scalar min over the W lanes of such a fully
  shuffled vector gives the min of all elements. Then the per-axis answer is
  min over lanes L with L % 3 == axis. This is only correct per axis when the
  lane->axis map is stable, which it is: L % 3 == (3p + a) % 3 == a % 3, and
  3p % W == 3p % 8 cycles, so lane L holds axis L % 3 always (since 3p % 8
  takes each residue exactly 3 times across 8 consecutive points). Kept as a
  note only: a2 is evaluated by measuring, not by reasoning.
"""
