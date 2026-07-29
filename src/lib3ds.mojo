"""Compute kernels derived from lib3ds, exposed through a flat C ABI."""

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


# lib3ds: lib3ds/vector.c lib3ds_vector_normalize
@always_inline
def normalize(x: Float32, y: Float32, z: Float32) -> SIMD[DType.float32, 3]:
    var length = sqrt(x * x + y * y + z * z)
    if abs(length) < 1.0e-8:
        if x >= y and x >= z:
            return SIMD[DType.float32, 3](1.0, 0.0, 0.0)
        if y >= z:
            return SIMD[DType.float32, 3](0.0, 1.0, 0.0)
        return SIMD[DType.float32, 3](0.0, 0.0, 1.0)
    var inverse = 1.0 / length
    return SIMD[DType.float32, 3](x * inverse, y * inverse, z * inverse)


# lib3ds: lib3ds/mesh.c lib3ds_mesh_calculate_normals
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
            py * qz - pz * qy,
            pz * qx - px * qz,
            px * qy - py * qx,
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
                    # Upstream keeps an unbounded linked list of normals already
                    # accumulated for this corner. Re-scan prior incident faces
                    # instead of imposing a fixed-degree scratch-buffer limit.
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


# lib3ds: lib3ds/mesh.c lib3ds_mesh_bounding_box
@export("m3ds_mesh_bounding_box")
def mesh_bounding_box(
    vertices_address: Int, point_count: Int, result_address: Int
) abi("C"):
    var vertices = f32p(vertices_address)
    var result = f32p(result_address)
    result[0] = 3.402823466e38
    result[1] = 3.402823466e38
    result[2] = 3.402823466e38
    # FLT_MIN is intentional: this preserves lib3ds's all-negative-mesh behavior.
    result[3] = 1.175494351e-38
    result[4] = 1.175494351e-38
    result[5] = 1.175494351e-38
    for point in range(point_count):
        for axis in range(3):
            var value = vertices[3 * point + axis]
            result[axis] = min(result[axis], value)
            result[3 + axis] = max(result[3 + axis], value)


# lib3ds: lib3ds/vector.c lib3ds_vector_transform
@always_inline
def transform_point_range(
    vertices: F32Ptr,
    matrix: F32Ptr,
    result: F32Ptr,
    start: Int,
    end: Int,
):
    comptime W = simd_width_of[DType.float64]()
    var vector_end = start + (end - start) // W * W
    for point in range(start, vector_end, W):
        var base = 3 * point
        var x = (vertices + base).strided_load[width=W](3)
        var y = (vertices + base + 1).strided_load[width=W](3)
        var z = (vertices + base + 2).strided_load[width=W](3)
        (result + base).strided_store[width=W](
            x * matrix[0] + y * matrix[4] + z * matrix[8] + matrix[12], 3
        )
        (result + base + 1).strided_store[width=W](
            x * matrix[1] + y * matrix[5] + z * matrix[9] + matrix[13], 3
        )
        (result + base + 2).strided_store[width=W](
            x * matrix[2] + y * matrix[6] + z * matrix[10] + matrix[14], 3
        )
    for point in range(vector_end, end):
        var x = vertices[3 * point]
        var y = vertices[3 * point + 1]
        var z = vertices[3 * point + 2]
        result[3 * point] = (
            matrix[0] * x + matrix[4] * y + matrix[8] * z + matrix[12]
        )
        result[3 * point + 1] = (
            matrix[1] * x + matrix[5] * y + matrix[9] * z + matrix[13]
        )
        result[3 * point + 2] = (
            matrix[2] * x + matrix[6] * y + matrix[10] * z + matrix[14]
        )


@export("m3ds_transform_points")
def transform_points(
    vertices_address: Int,
    matrix_address: Int,
    result_address: Int,
    point_count: Int,
) abi("C"):
    var vertices = f32p(vertices_address)
    var matrix = f32p(matrix_address)
    var result = f32p(result_address)
    transform_point_range(vertices, matrix, result, 0, point_count)


@always_inline
def matrix_multiply(
    left: F32Ptr,
    right: InlineArray[Float32, 16],
    destination: F32Ptr,
):
    var temporary = InlineArray[Float32, 16](fill=0.0)
    for column in range(4):
        for row in range(4):
            for k in range(4):
                temporary[4 * column + row] += (
                    left[4 * k + row] * right[4 * column + k]
                )
    for index in range(16):
        destination[index] = temporary[index]


# lib3ds: lib3ds/node.c lib3ds_node_eval
@export("m3ds_compose_nodes")
def compose_nodes(
    parents_address: Int,
    types_address: Int,
    positions_address: Int,
    rotations_address: Int,
    scales_address: Int,
    matrices_address: Int,
    node_count: Int,
) abi("C") -> Int:
    var parents = i64p(parents_address)
    var types = i64p(types_address)
    var positions = f32p(positions_address)
    var rotations = f32p(rotations_address)
    var scales = f32p(scales_address)
    var matrices = f32p(matrices_address)
    for node in range(node_count):
        var parent = Int(parents[node])
        if parent >= node or parent < -1:
            return 0
        var local = InlineArray[Float32, 16](fill=0.0)
        local[0] = 1.0
        local[5] = 1.0
        local[10] = 1.0
        local[15] = 1.0
        local[12] = positions[3 * node]
        local[13] = positions[3 * node + 1]
        local[14] = positions[3 * node + 2]
        if types[node] == 2:
            var x = rotations[4 * node]
            var y = rotations[4 * node + 1]
            var z = rotations[4 * node + 2]
            var w = rotations[4 * node + 3]
            var length2 = x * x + y * y + z * z + w * w
            var factor: Float32 = 1.0 if abs(length2) < 1.0e-8 else 2.0 / length2
            var xs = x * factor
            var ys = y * factor
            var zs = z * factor
            local[0] = (1.0 - (y * ys + z * zs)) * scales[3 * node]
            local[1] = (x * ys + w * zs) * scales[3 * node]
            local[2] = (x * zs - w * ys) * scales[3 * node]
            local[4] = (x * ys - w * zs) * scales[3 * node + 1]
            local[5] = (1.0 - (x * xs + z * zs)) * scales[3 * node + 1]
            local[6] = (y * zs + w * xs) * scales[3 * node + 1]
            local[8] = (x * zs + w * ys) * scales[3 * node + 2]
            local[9] = (y * zs - w * xs) * scales[3 * node + 2]
            local[10] = (1.0 - (x * xs + y * ys)) * scales[3 * node + 2]
        var destination = matrices + 16 * node
        if parent < 0:
            for index in range(16):
                destination[index] = local[index]
        else:
            matrix_multiply(matrices + 16 * parent, local, destination)
    return 1


# lib3ds: lib3ds/ease.c lib3ds_ease
@export("m3ds_ease")
def ease(
    previous: Float32,
    current: Float32,
    following: Float32,
    ease_from: Float32,
    ease_to: Float32,
) abi("C") -> Float32:
    var step = (current - previous) / (following - previous)
    var value = step
    var total = ease_to + ease_from
    var ef = ease_from
    var et = ease_to
    if total != 0.0:
        if total > 1.0:
            et /= total
            ef /= total
        var coefficient = 1.0 / (2.0 - (et + ef))
        if step < ef:
            value = coefficient / ef * step * step
        elif 1.0 - et <= step:
            step = 1.0 - step
            value = 1.0 - coefficient / et * step * step
        else:
            value = (2.0 * step - ef) * coefficient
    return value
