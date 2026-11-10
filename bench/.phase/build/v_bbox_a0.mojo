"""Development arm B: SIMD bounding box (base copied from src/lib3ds.mojo)."""

from std.math import sqrt
from std.sys import simd_width_of

comptime F32Ptr = UnsafePointer[Float32, AnyOrigin[mut=True]]
comptime I64Ptr = UnsafePointer[Int64, AnyOrigin[mut=True]]
comptime U32Ptr = UnsafePointer[UInt32, AnyOrigin[mut=True]]


def f32p(address: Int) -> F32Ptr:
    return F32Ptr(unsafe_from_address=address)


def i64p(address: Int) -> I64Ptr:
    return I64Ptr(unsafe_from_address=address)


def u32p(address: Int) -> U32Ptr:
    return U32Ptr(unsafe_from_address=address)


@always_inline
def normalize(x: Float32, y: Float32, z: Float32) -> SIMD[DType.float32, 4]:
    var length = sqrt(x * x + y * y + z * z)
    if abs(length) < 1.0e-8:
        if x >= y and x >= z:
            return SIMD[DType.float32, 4](1.0, 0.0, 0.0, 0.0)
        if y >= z:
            return SIMD[DType.float32, 4](0.0, 1.0, 0.0, 0.0)
        return SIMD[DType.float32, 4](0.0, 0.0, 1.0, 0.0)
    var inverse = 1.0 / length
    return SIMD[DType.float32, 4](x * inverse, y * inverse, z * inverse, 0.0)


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
    var vertices = f32p(vertices_address)
    var faces = i64p(faces_address)
    var smoothing = u32p(smoothing_address)
    var offsets = i64p(offsets_address)
    var adjacent = i64p(adjacent_address)
    var face_normals = f32p(face_normals_address)
    var result = f32p(result_address)
    for face in range(face_count):
        var ia = Int(faces[3 * face])
        var ib = Int(faces[3 * face + 1])
        var ic = Int(faces[3 * face + 2])
        var px = vertices[3 * ic] - vertices[3 * ib]
        var py = vertices[3 * ic + 1] - vertices[3 * ib + 1]
        var pz = vertices[3 * ic + 2] - vertices[3 * ib + 2]
        var qx = vertices[3 * ia] - vertices[3 * ib]
        var qy = vertices[3 * ia + 1] - vertices[3 * ib + 1]
        var qz = vertices[3 * ia + 2] - vertices[3 * ib + 2]
        var normal = normalize(
            py * qz - pz * qy, pz * qx - px * qz, px * qy - py * qx
        )
        face_normals[3 * face] = normal[0]
        face_normals[3 * face + 1] = normal[1]
        face_normals[3 * face + 2] = normal[2]
    for face in range(face_count):
        for corner in range(3):
            var nx: Float32 = 0.0
            var ny: Float32 = 0.0
            var nz: Float32 = 0.0
            if smoothing[face] == 0:
                nx = face_normals[3 * face]
                ny = face_normals[3 * face + 1]
                nz = face_normals[3 * face + 2]
            else:
                var vertex = Int(faces[3 * face + corner])
                var cursor = Int(offsets[vertex])
                var adjacency_end = Int(offsets[vertex + 1])
                while cursor < adjacency_end:
                    var other = Int(adjacent[cursor])
                    var ox = face_normals[3 * other]
                    var oy = face_normals[3 * other + 1]
                    var oz = face_normals[3 * other + 2]
                    var found = False
                    var prior = Int(offsets[vertex])
                    while prior < cursor:
                        var prior_face = Int(adjacent[prior])
                        if (smoothing[face] & smoothing[prior_face]) != 0:
                            if abs(
                                face_normals[3 * prior_face] * ox
                                + face_normals[3 * prior_face + 1] * oy
                                + face_normals[3 * prior_face + 2] * oz
                                - 1.0
                            ) < 1.0e-5:
                                found = True
                                break
                        prior += 1
                    if not found and (smoothing[face] & smoothing[other]) != 0:
                        nx += ox
                        ny += oy
                        nz += oz
                    cursor += 1
            var normal = normalize(nx, ny, nz)
            var destination = 9 * face + 3 * corner
            result[destination] = normal[0]
            result[destination + 1] = normal[1]
            result[destination + 2] = normal[2]



# CANDIDATE a0: transposed 3xW loads, one min/max per load, W-wide accumulators.
@always_inline
def bounding_box_kernel(vertices_address: Int, point_count: Int, result_address: Int):
    var vertices = f32p(vertices_address)
    var result = f32p(result_address)
    comptime W = simd_width_of[DType.float32]()
    comptime V = 3 * W
    comptime LOW = W // 2
    comptime HIGH = W - LOW
    var minimum = SIMD[DType.float32, W](3.402823466e38)
    var maximum = SIMD[DType.float32, W](1.175494351e-38)
    var i = 0
    while i + 2 * V <= 3 * point_count:
        var a0 = vertices.load[width=W](i)
        var a1 = vertices.load[width=W](i + W)
        var a2 = vertices.load[width=W](i + 2 * W)
        var b0 = vertices.load[width=W](i + V)
        var b1 = vertices.load[width=W](i + V + W)
        var b2 = vertices.load[width=W](i + V + 2 * W)
        minimum = min(minimum, min(min(a0, a1), a2))
        minimum = min(minimum, min(min(b0, b1), b2))
        maximum = max(maximum, max(max(a0, a1), a2))
        maximum = max(maximum, max(max(b0, b1), b2))
        i += 2 * V
    while i + V <= 3 * point_count:
        var a0 = vertices.load[width=W](i)
        var a1 = vertices.load[width=W](i + W)
        var a2 = vertices.load[width=W](i + 2 * W)
        minimum = min(minimum, min(min(a0, a1), a2))
        maximum = max(maximum, max(max(a0, a1), a2))
        i += V
    # (x0,y0,z0,x1,y1,z1,...) -> x lanes are i % 3 == 0, etc. Lane 0 is x.
    var min_half = SIMD[DType.float32, LOW](minimum)
    var max_half = SIMD[DType.float32, HIGH](maximum)
    var min_lo = minimum.extract[0:LOW]()
    var min_hi = minimum.extract[LOW:W]()
    var max_lo = maximum.extract[0:LOW]()
    var max_hi = maximum.extract[LOW:W]()
    minimum = min(min_lo.join(min_hi), min_half)
    maximum = max(max_lo.join(max_hi), max_half)
    var low = minimum.extract[0:LOW]()
    var high = minimum.extract[LOW:W]()
    # x = min of 2 lanes, y = min of 2 lanes, z = min of 1 lane: reduce
    # (the last lane is a duplicate of x for W=8; harmless).
    var x = min(low[0], high[0])
    var y = min(low[1], high[1])
    var z = low[2]
    var max_x = max(max_lo[0], max_hi[0])
    var max_y = max(max_lo[1], max_hi[1])
    var max_z = max_lo[2]
    # scalar tail (guarded so min-only meshes keep the FLT_MIN initialiser)
    for point in range(i // 3, point_count):
        var base = 3 * point
        var px = vertices[base]
        var py = vertices[base + 1]
        var pz = vertices[base + 2]
        x = min(x, px)
        y = min(y, py)
        z = min(z, pz)
        max_x = max(max_x, px)
        max_y = max(max_y, py)
        max_z = max(max_z, pz)
    result[0] = x
    result[1] = y
    result[2] = z
    result[3] = max_x
    result[4] = max_y
    result[5] = max_z


# ---- development probe wrappers, generated by bench/.phase/gen_probe.py ----

@export("probe_bounding_box")
def probe_bounding_box(address: Int, point_count: Int, result_address: Int) abi("C"):
    bounding_box_kernel(address, point_count, result_address)


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
