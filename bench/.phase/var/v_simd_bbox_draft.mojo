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


@export("m3ds_transform_points")
def transform_points(
    vertices_address: Int,
    matrix_address: Int,
    result_address: Int,
    point_count: Int,
) abi("C"):
    pass


# CANDIDATE: SIMD bounding box.
# 48 floats per block (16 points) = 8 vectors on AVX-512, 16 on AVX2.
# Per-lane (x, y, z) accumulators joined with two shuffles, then one
# horizontal reduce. The scalar tail alone preserves lib3ds's FLT_MIN
# max-initialiser semantics when the mesh is smaller than one block.
@always_inline
def bounding_box_kernel(vertices_address: Int, point_count: Int, result_address: Int):
    var vertices = f32p(vertices_address)
    var result = f32p(result_address)
    comptime W = simd_width_of[DType.float32]()
    comptime V = 3 * W
    comptime LOW = W // 2
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
        var c0 = a0.join(b0)
        var c1 = a1.join(b1)
        var c2 = a2.join(b2)
        minimum = min(
            minimum,
            min(a0, min(a1, a2)),
            min(b0, min(b1, b2)),
        )
        maximum = max(
            maximum,
            min(c0, min(c1, c2)),
        )
        i += 2 * V
    # max upper half explicitly: mixing min/max lanes stays exact
    var maxv = maximum
    var minv = minimum
    while i + V <= 3 * point_count:
        maxv = max(maxv, min(vertices.load[width=W](i), min(
            vertices.load[width=W](i + W), vertices.load[width=W](i + 2 * W))))
        i += V
    var lo = SIMD[DType.float32, LOW].cast[DType.float32]() if False else maxv
    result[0] = 0.0
