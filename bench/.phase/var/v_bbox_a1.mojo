"""Dev arm: bounding-box candidates measured against src (see bench/.phase/ab.py)."""

from std.sys import simd_width_of

comptime F32Ptr = UnsafePointer[Float32, AnyOrigin[mut=True]]


def f32p(address: Int) -> F32Ptr:
    return F32Ptr(unsafe_from_address=address)


# --- unchanged export from src, kept so the file is a full compile unit ---

@export("m3ds_mesh_normals")
def mesh_normals(
    vertices_address: Int,
    faces_address: Int,
    smoothing_address: Int,
    offsets_address: Int,
    adjacent_address: Int,
    face_normals_address: Int,
    result_address: Int,
    face_count: Int,
) abi("C"):
    pass


# CEILING: contiguous 8-wide min/max of every float, axes not separated.
# Not a shippable kernel; it bounds what any load-side vectorisation can reach.
@always_inline
def _bbox_ceiling(vertices_address: Int, point_count: Int, result_address: Int):
    var vertices = f32p(vertices_address)
    var result = f32p(result_address)
    comptime W = simd_width_of[DType.float32]()
    var minimum = SIMD[DType.float32, W](3.402823466e38)
    var maximum = SIMD[DType.float32, W](1.175494351e-38)
    var total = 3 * point_count
    var index = 0
    while index + W <= total:
        var v = vertices.load[width=W](index)
        minimum = min(minimum, v)
        maximum = max(maximum, v)
        index += W
    var lo = minimum.reduce_min()
    var hi = maximum.reduce_max()
    while index < total:
        lo = min(lo, vertices.unsafe_load(index))
        hi = max(hi, vertices.unsafe_load(index))
        index += 1
    result[0] = lo
    result[1] = lo
    result[2] = lo
    result[3] = hi
    result[4] = hi
    result[5] = hi


# CANDIDATE a1: per-axis strided loads (same access shape as the transform
# kernel), one accumulator per axis, three horizontal reductions after the loop.
@always_inline
def _bbox_a1(vertices_address: Int, point_count: Int, result_address: Int):
    var vertices = f32p(vertices_address)
    var result = f32p(result_address)
    comptime W = simd_width_of[DType.float32]()
    comptime LOW = W // 2
    comptime HIGH = W - LOW
    var min_x = SIMD[DType.float32, W](3.402823466e38)
    var min_y = SIMD[DType.float32, W](3.402823466e38)
    var min_z = SIMD[DType.float32, W](3.402823466e38)
    var max_x = SIMD[DType.float32, W](1.175494351e-38)
    var max_y = SIMD[DType.float32, W](1.175494351e-38)
    var max_z = SIMD[DType.float32, W](1.175494351e-38)
    var end = point_count // W * W
    var point = 0
    while point < end:
        var base = 3 * point
        var x = (vertices + base).strided_load[3, width=W]
        var y = (vertices + base + 1).strided_load[3, width=W]
        var z = (vertices + base + 2).strided_load[3, width=W]
        min_x = min(min_x, x)
        min_y = min(min_y, y)
        min_z = min(min_z, z)
        max_x = max(max_x, x)
        max_y = max(max_y, y)
        max_z = max(max_z, z)
        point += W
    var lo_x = min_x.reduce_min()
    var lo_y = min_y.reduce_min()
    var lo_z = min_z.reduce_min()
    var hi_x = max_x.reduce_max()
    var hi_y = max_y.reduce_max()
    var hi_z = max_z.reduce_max()
    while point < point_count:
        var base = 3 * point
        var x = vertices.unsafe_load(base)
        var y = vertices.unsafe_load(base + 1)
        var z = vertices.unsafe_load(base + 2)
        lo_x = min(lo_x, x)
        lo_y = min(lo_y, y)
        lo_z = min(lo_z, z)
        hi_x = max(hi_x, x)
        hi_y = max(hi_y, y)
        hi_z = max(hi_z, z)
        point += 1
    result[0] = lo_x
    result[1] = lo_y
    result[2] = lo_z
    result[3] = hi_x
    result[4] = hi_y
    result[5] = hi_z


# CANDIDATE a2: contiguous loads only. 24 floats (8 points) at a time.
# Axes are separated by pre-shuffling the three vectors with the fixed lane
# permutation that groups x lanes, then y, then z; see bench/analysis.md.
@always_inline
def _bbox_a2(vertices_address: Int, point_count: Int, result_address: Int):
    var vertices = f32p(vertices_address)
    var result = f32p(result_address)
    comptime W = simd_width_of[DType.float32]()
    var min_x = SIMD[DType.float32, W](3.402823466e38)
    var min_y = SIMD[DType.float32, W](3.402823466e38)
    var min_z = SIMD[DType.float32, W](3.402823466e38)
    var max_x = SIMD[DType.float32, W](1.175494351e-38)
    var max_y = SIMD[DType.float32, W](1.175494351e-38)
    var max_z = SIMD[DType.float32, W](1.175494351e-38)
    var block = 3 * W
    var end = 3 * (point_count // W * W)
    var index = 0
    while index < end:
        var a = vertices.load[width=W](index)
        var b = vertices.load[width=W](index + W)
        var c = vertices.load[width=W](index + 2 * W)
        # rotate the triple so each output vector holds one axis in 8 lanes
        var x = SIMD[DType.float32, W](
            a[0], a[3], a[6], b[1], b[4], b[7], c[2], c[5]
        )
        var y = SIMD[DType.float32, W](
            a[1], a[4], a[7], b[2], b[5], c[0], c[3], c[6]
        )
        var z = SIMD[DType.float32, W](
            a[2], a[5], b[0], b[3], b[6], c[1], c[4], c[7]
        )
        min_x = min(min_x, x)
        min_y = min(min_y, y)
        min_z = min(min_z, z)
        max_x = max(max_x, x)
        max_y = max(max_y, y)
        max_z = max(max_z, z)
        index += block
    var lo_x = min_x.reduce_min()
    var lo_y = min_y.reduce_min()
    var lo_z = min_z.reduce_min()
    var hi_x = max_x.reduce_max()
    var hi_y = max_y.reduce_max()
    var hi_z = max_z.reduce_max()
    var point = index // 3
    while point < point_count:
        var base = 3 * point
        var x = vertices.unsafe_load(base)
        var y = vertices.unsafe_load(base + 1)
        var z = vertices.unsafe_load(base + 2)
        lo_x = min(lo_x, x)
        lo_y = min(lo_y, y)
        lo_z = min(lo_z, z)
        hi_x = max(hi_x, x)
        hi_y = max(hi_y, y)
        hi_z = max(hi_z, z)
        point += 1
    result[0] = lo_x
    result[1] = lo_y
    result[2] = lo_z
    result[3] = hi_x
    result[4] = hi_y
    result[5] = hi_z


@export("m3ds_mesh_bounding_box")
def mesh_bounding_box(
    vertices_address: Int, point_count: Int, result_address: Int
) abi("C"):
    _bbox_a1(vertices_address, point_count, result_address)
