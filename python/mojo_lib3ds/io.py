from __future__ import annotations

import math
import struct
from collections.abc import Iterator

import numpy as np

from .model import Camera, Key, Light, Material, Mesh, Node, Scene, TextureMap, Track

M3DMAGIC = 0x4D4D
M3D_VERSION = 0x0002
COLOR_F, COLOR_24, LIN_COLOR_24, LIN_COLOR_F = 0x10, 0x11, 0x12, 0x13
INT_PERCENTAGE, FLOAT_PERCENTAGE = 0x30, 0x31
MDATA, MESH_VERSION, MASTER_SCALE, AMBIENT_LIGHT = 0x3D3D, 0x3D3E, 0x0100, 0x2100
MAT_ENTRY, MAT_NAME = 0xAFFF, 0xA000
MAT_AMBIENT, MAT_DIFFUSE, MAT_SPECULAR = 0xA010, 0xA020, 0xA030
MAT_SHININESS, MAT_SHIN2PCT, MAT_TRANSPARENCY = 0xA040, 0xA041, 0xA050
MAT_SHADING, MAT_SELF_ILLUM, MAT_TWO_SIDE = 0xA100, 0xA080, 0xA081
MAT_WIRE, MAT_WIRE_SIZE, MAT_TEXMAP = 0xA085, 0xA087, 0xA200
MAT_MAPNAME, MAT_MAP_TILING, MAT_MAP_TEXBLUR = 0xA300, 0xA351, 0xA353
MAT_MAP_USCALE, MAT_MAP_VSCALE = 0xA354, 0xA356
MAT_MAP_UOFFSET, MAT_MAP_VOFFSET, MAT_MAP_ANG = 0xA358, 0xA35A, 0xA35C
NAMED_OBJECT = 0x4000
OBJ_HIDDEN, OBJ_VIS_LOFTER, OBJ_DOESNT_CAST = 0x4010, 0x4011, 0x4012
OBJ_MATTE, OBJ_FAST, OBJ_FROZEN, OBJ_DONT_RCVSHADOW = (
    0x4013,
    0x4014,
    0x4016,
    0x4017,
)
N_TRI_OBJECT, POINT_ARRAY, POINT_FLAG_ARRAY, FACE_ARRAY = (
    0x4100,
    0x4110,
    0x4111,
    0x4120,
)
MSH_MAT_GROUP, SMOOTH_GROUP, TEX_VERTS = 0x4130, 0x4150, 0x4140
MESH_MATRIX, MESH_COLOR = 0x4160, 0x4165
N_DIRECT_LIGHT, DL_SPOTLIGHT = 0x4600, 0x4610
DL_OFF, DL_ATTENUATE, DL_SHADOWED = 0x4620, 0x4625, 0x4630
DL_SEE_CONE, DL_SPOT_RECTANGULAR = 0x4650, 0x4651
DL_INNER_RANGE, DL_OUTER_RANGE, DL_MULTIPLIER = 0x4659, 0x465A, 0x465B
DL_SPOT_ROLL = 0x4656
N_CAMERA, CAM_SEE_CONE, CAM_RANGES = 0x4700, 0x4710, 0x4720
KFDATA, KFHDR, KFSEG, KFCURTIME = 0xB000, 0xB00A, 0xB008, 0xB009
NODE_TAGS = {
    0xB001: 1,
    0xB002: 2,
    0xB003: 3,
    0xB004: 4,
    0xB005: 5,
    0xB006: 6,
    0xB007: 5,
}
TYPE_TAGS = {1: 0xB001, 2: 0xB002, 3: 0xB003, 4: 0xB004, 5: 0xB005, 6: 0xB006}
NODE_ID, NODE_HDR, PIVOT, INSTANCE_NAME = 0xB030, 0xB010, 0xB013, 0xB011
POS_TRACK, ROT_TRACK, SCL_TRACK = 0xB020, 0xB021, 0xB022
COL_TRACK, FOV_TRACK, ROLL_TRACK = 0xB025, 0xB023, 0xB024

_MIN_PAYLOAD = {
    COLOR_F: 12,
    COLOR_24: 3,
    LIN_COLOR_F: 12,
    LIN_COLOR_24: 3,
    INT_PERCENTAGE: 2,
    FLOAT_PERCENTAGE: 4,
    MESH_VERSION: 4,
    MASTER_SCALE: 4,
    MAT_SHADING: 2,
    MAT_WIRE_SIZE: 4,
    MAT_MAP_TILING: 2,
    MAT_MAP_TEXBLUR: 4,
    MAT_MAP_USCALE: 4,
    MAT_MAP_VSCALE: 4,
    MAT_MAP_UOFFSET: 4,
    MAT_MAP_VOFFSET: 4,
    MAT_MAP_ANG: 4,
    POINT_ARRAY: 2,
    FACE_ARRAY: 2,
    TEX_VERTS: 2,
    MESH_MATRIX: 48,
    MESH_COLOR: 1,
    N_CAMERA: 32,
    CAM_RANGES: 8,
    N_DIRECT_LIGHT: 12,
    DL_INNER_RANGE: 4,
    DL_OUTER_RANGE: 4,
    DL_MULTIPLIER: 4,
    DL_SPOTLIGHT: 20,
    DL_SPOT_ROLL: 4,
    NODE_ID: 2,
    PIVOT: 12,
    POS_TRACK: 14,
    ROT_TRACK: 14,
    SCL_TRACK: 14,
    KFHDR: 2,
    KFSEG: 8,
    KFCURTIME: 4,
}


class FormatError(ValueError):
    pass


def _chunk(chunk_id: int, payload: bytes = b"") -> bytes:
    return struct.pack("<HI", chunk_id, len(payload) + 6) + payload


def _cstring(value: str) -> bytes:
    encoded = value.encode("latin-1")
    if b"\0" in encoded:
        raise ValueError("3DS strings cannot contain NUL")
    return encoded + b"\0"


def _read_string(data: memoryview, start: int, end: int) -> tuple[str, int]:
    raw = data[start:end].tobytes()
    terminator = raw.find(b"\0")
    if terminator < 0:
        raise FormatError("unterminated 3DS string")
    return raw[:terminator].decode("latin-1"), start + terminator + 1


# lib3ds: lib3ds/chunk.c lib3ds_chunk_read_next
def _chunks(data: memoryview, start: int, end: int) -> Iterator[tuple[int, int, int]]:
    cursor = start
    while cursor < end:
        if end - cursor < 6:
            raise FormatError("truncated chunk header")
        chunk_id, size = struct.unpack_from("<HI", data, cursor)
        if size < 6 or cursor + size > end:
            raise FormatError(f"invalid chunk size {size} at byte {cursor}")
        payload_size = size - 6
        if payload_size < _MIN_PAYLOAD.get(chunk_id, 0):
            raise FormatError(f"truncated 0x{chunk_id:04X} chunk")
        yield chunk_id, cursor + 6, cursor + size
        cursor += size


def _vector(data: memoryview, offset: int) -> np.ndarray:
    return np.array(struct.unpack_from("<3f", data, offset), dtype=np.float32)


def _read_color(data: memoryview, start: int, end: int) -> np.ndarray:
    color = np.zeros(3, dtype=np.float32)
    have_linear = False
    for chunk_id, payload, chunk_end in _chunks(data, start, end):
        if chunk_id in (LIN_COLOR_24, COLOR_24):
            if chunk_id == LIN_COLOR_24 or not have_linear:
                color[:] = np.frombuffer(data[payload : payload + 3], dtype=np.uint8) / 255
            have_linear |= chunk_id == LIN_COLOR_24
        elif chunk_id in (LIN_COLOR_F, COLOR_F):
            if chunk_id == LIN_COLOR_F or not have_linear:
                color[:] = struct.unpack_from("<3f", data, payload)
            have_linear |= chunk_id == LIN_COLOR_F
    return color


def _read_percentage(data: memoryview, start: int, end: int) -> float:
    value = 0.0
    for chunk_id, payload, _ in _chunks(data, start, end):
        if chunk_id == INT_PERCENTAGE:
            value = struct.unpack_from("<h", data, payload)[0] / 100.0
        elif chunk_id == FLOAT_PERCENTAGE:
            value = struct.unpack_from("<f", data, payload)[0]
    return value


def _read_texture(data: memoryview, start: int, end: int) -> TextureMap:
    texture = TextureMap()
    for chunk_id, payload, chunk_end in _chunks(data, start, end):
        if chunk_id == INT_PERCENTAGE:
            texture.percent = struct.unpack_from("<h", data, payload)[0] / 100.0
        elif chunk_id == MAT_MAPNAME:
            texture.name, _ = _read_string(data, payload, chunk_end)
        elif chunk_id == MAT_MAP_TILING:
            texture.flags = struct.unpack_from("<H", data, payload)[0]
        elif chunk_id == MAT_MAP_TEXBLUR:
            texture.blur = struct.unpack_from("<f", data, payload)[0]
        elif chunk_id in (MAT_MAP_USCALE, MAT_MAP_VSCALE):
            texture.scale[int(chunk_id == MAT_MAP_VSCALE)] = struct.unpack_from(
                "<f", data, payload
            )[0]
        elif chunk_id in (MAT_MAP_UOFFSET, MAT_MAP_VOFFSET):
            texture.offset[int(chunk_id == MAT_MAP_VOFFSET)] = struct.unpack_from(
                "<f", data, payload
            )[0]
        elif chunk_id == MAT_MAP_ANG:
            texture.rotation = struct.unpack_from("<f", data, payload)[0]
    return texture


def _read_material(data: memoryview, start: int, end: int) -> Material:
    material = Material()
    for chunk_id, payload, chunk_end in _chunks(data, start, end):
        if chunk_id == MAT_NAME:
            material.name, _ = _read_string(data, payload, chunk_end)
        elif chunk_id in (MAT_AMBIENT, MAT_DIFFUSE, MAT_SPECULAR):
            value = _read_color(data, payload, chunk_end)
            setattr(
                material,
                {
                    MAT_AMBIENT: "ambient",
                    MAT_DIFFUSE: "diffuse",
                    MAT_SPECULAR: "specular",
                }[chunk_id],
                value,
            )
        elif chunk_id in (MAT_SHININESS, MAT_SHIN2PCT, MAT_TRANSPARENCY):
            setattr(
                material,
                {
                    MAT_SHININESS: "shininess",
                    MAT_SHIN2PCT: "shin_strength",
                    MAT_TRANSPARENCY: "transparency",
                }[chunk_id],
                _read_percentage(data, payload, chunk_end),
            )
        elif chunk_id == MAT_SHADING:
            material.shading = struct.unpack_from("<h", data, payload)[0]
        elif chunk_id == MAT_SELF_ILLUM:
            material.self_illum = True
        elif chunk_id == MAT_TWO_SIDE:
            material.two_sided = True
        elif chunk_id == MAT_WIRE:
            material.use_wire = True
        elif chunk_id == MAT_WIRE_SIZE:
            material.wire_size = struct.unpack_from("<f", data, payload)[0]
        elif chunk_id == MAT_TEXMAP:
            material.texture1_map = _read_texture(data, payload, chunk_end)
    return material


def _read_faces(
    data: memoryview, start: int, end: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str]]:
    if end - start < 2:
        raise FormatError("truncated face array")
    count = struct.unpack_from("<H", data, start)[0]
    base_end = start + 2 + 8 * count
    if base_end > end:
        raise FormatError("truncated face records")
    records = np.frombuffer(data[start + 2 : base_end], dtype="<u2").reshape(count, 4)
    faces = np.ascontiguousarray(records[:, :3], dtype=np.int64)
    flags = np.ascontiguousarray(records[:, 3], dtype=np.uint16)
    smoothing = np.zeros(count, dtype=np.uint32)
    materials = [""] * count
    for chunk_id, payload, chunk_end in _chunks(data, base_end, end):
        if chunk_id == SMOOTH_GROUP:
            if chunk_end - payload < 4 * count:
                raise FormatError("truncated smoothing groups")
            smoothing[:] = np.frombuffer(
                data[payload : payload + 4 * count], dtype="<u4"
            )
        elif chunk_id == MSH_MAT_GROUP:
            name, cursor = _read_string(data, payload, chunk_end)
            if cursor + 2 > chunk_end:
                raise FormatError("truncated material face group count")
            number = struct.unpack_from("<H", data, cursor)[0]
            cursor += 2
            if cursor + 2 * number > chunk_end:
                raise FormatError("truncated material face group")
            for index in struct.unpack_from(f"<{number}H", data, cursor):
                if index >= count:
                    raise FormatError("material group face index out of range")
                materials[index] = name
    return faces, flags, smoothing, materials


def _read_mesh(data: memoryview, start: int, end: int, name: str) -> Mesh:
    vertices = np.empty((0, 3), dtype=np.float32)
    faces = np.empty((0, 3), dtype=np.int64)
    flags = np.empty(0, dtype=np.uint16)
    smoothing = np.empty(0, dtype=np.uint32)
    materials: list[str] = []
    texcoords = None
    matrix = np.ascontiguousarray(np.eye(4), dtype=np.float32)
    color = 0
    for chunk_id, payload, chunk_end in _chunks(data, start, end):
        if chunk_id == POINT_ARRAY:
            count = struct.unpack_from("<H", data, payload)[0]
            required = payload + 2 + 12 * count
            if required > chunk_end:
                raise FormatError("truncated point array")
            vertices = np.frombuffer(
                data[payload + 2 : required], dtype="<f4"
            ).reshape(count, 3).copy()
        elif chunk_id == FACE_ARRAY:
            faces, flags, smoothing, materials = _read_faces(
                data, payload, chunk_end
            )
        elif chunk_id == TEX_VERTS:
            count = struct.unpack_from("<H", data, payload)[0]
            required = payload + 2 + 8 * count
            if required > chunk_end:
                raise FormatError("truncated texture coordinates")
            texcoords = np.frombuffer(
                data[payload + 2 : required], dtype="<f4"
            ).reshape(count, 2).copy()
        elif chunk_id == MESH_MATRIX:
            if chunk_end - payload < 48:
                raise FormatError("truncated mesh matrix")
            values = struct.unpack_from("<12f", data, payload)
            for column in range(4):
                matrix[column, :3] = values[3 * column : 3 * column + 3]
        elif chunk_id == MESH_COLOR:
            color = data[payload]
    mesh = Mesh(
        name,
        vertices,
        faces,
        smoothing,
        flags,
        materials,
        texcoords,
        matrix,
        color=color,
    )
    # lib3ds: lib3ds/mesh.c lib3ds_mesh_read negative-determinant correction
    if np.linalg.det(matrix) < 0.0 and len(vertices):
        inverse = np.linalg.inv(matrix)
        scaled = matrix.copy()
        scaled[0, :] *= -1.0
        correction = inverse @ scaled
        mesh.vertices = mesh.transformed_vertices(
            np.ascontiguousarray(correction, dtype=np.float32)
        )
    return mesh


def _read_camera(data: memoryview, start: int, end: int, name: str) -> Camera:
    if end - start < 32:
        raise FormatError("truncated camera")
    position = _vector(data, start)
    target = _vector(data, start + 12)
    roll, lens = struct.unpack_from("<2f", data, start + 24)
    camera = Camera(
        name,
        position,
        target,
        roll,
        45.0 if abs(lens) < 1.0e-8 else 2400.0 / lens,
    )
    for chunk_id, payload, _ in _chunks(data, start + 32, end):
        if chunk_id == CAM_SEE_CONE:
            camera.see_cone = True
        elif chunk_id == CAM_RANGES:
            camera.near_range, camera.far_range = struct.unpack_from(
                "<2f", data, payload
            )
    return camera


def _read_spotlight(data: memoryview, start: int, end: int, light: Light) -> None:
    if end - start < 20:
        raise FormatError("truncated spotlight")
    light.spot_light = True
    light.spot = _vector(data, start)
    light.hot_spot, light.fall_off = struct.unpack_from("<2f", data, start + 12)
    for chunk_id, payload, _ in _chunks(data, start + 20, end):
        if chunk_id == DL_SPOT_ROLL:
            light.roll = struct.unpack_from("<f", data, payload)[0]
        elif chunk_id == DL_SHADOWED:
            light.shadowed = True
        elif chunk_id == DL_SEE_CONE:
            light.see_cone = True
        elif chunk_id == DL_SPOT_RECTANGULAR:
            light.rectangular_spot = True


def _read_light(data: memoryview, start: int, end: int, name: str) -> Light:
    if end - start < 12:
        raise FormatError("truncated light")
    light = Light(name, position=_vector(data, start))
    for chunk_id, payload, chunk_end in _chunks(data, start + 12, end):
        if chunk_id == COLOR_F:
            light.color = _vector(data, payload)
        elif chunk_id == DL_OFF:
            light.off = True
        elif chunk_id == DL_OUTER_RANGE:
            light.outer_range = struct.unpack_from("<f", data, payload)[0]
        elif chunk_id == DL_INNER_RANGE:
            light.inner_range = struct.unpack_from("<f", data, payload)[0]
        elif chunk_id == DL_MULTIPLIER:
            light.multiplier = struct.unpack_from("<f", data, payload)[0]
        elif chunk_id == DL_ATTENUATE:
            # lib3ds reads a float despite writing a zero-sized switch chunk.
            light.attenuation = (
                struct.unpack_from("<f", data, payload)[0]
                if chunk_end - payload >= 4
                else 1.0
            )
        elif chunk_id == DL_SPOTLIGHT:
            _read_spotlight(data, payload, chunk_end, light)
    return light


def _read_named_object(
    scene: Scene, data: memoryview, start: int, end: int
) -> None:
    name, cursor = _read_string(data, start, end)
    parsed: Mesh | Camera | Light | None = None
    object_flags = 0
    flag_chunks = {
        OBJ_HIDDEN: 0x01,
        OBJ_VIS_LOFTER: 0x02,
        OBJ_DOESNT_CAST: 0x04,
        OBJ_MATTE: 0x08,
        OBJ_DONT_RCVSHADOW: 0x10,
        OBJ_FAST: 0x20,
        OBJ_FROZEN: 0x40,
    }
    for chunk_id, payload, chunk_end in _chunks(data, cursor, end):
        if chunk_id == N_TRI_OBJECT:
            parsed = _read_mesh(data, payload, chunk_end, name)
            scene.meshes.append(parsed)
        elif chunk_id == N_CAMERA:
            parsed = _read_camera(data, payload, chunk_end, name)
            scene.cameras.append(parsed)
        elif chunk_id == N_DIRECT_LIGHT:
            parsed = _read_light(data, payload, chunk_end, name)
            scene.lights.append(parsed)
        elif chunk_id in flag_chunks:
            object_flags |= flag_chunks[chunk_id]
    if parsed is not None:
        parsed.object_flags = object_flags


def _axis_angle_quaternion(angle: float, axis: np.ndarray) -> np.ndarray:
    length = float(np.linalg.norm(axis))
    if length < 1.0e-8:
        return np.array([0, 0, 0, 1], dtype=np.float32)
    omega = -0.5 * angle
    return np.array(
        [*(axis * (math.sin(omega) / length)), math.cos(omega)], dtype=np.float32
    )


def _quat_multiply(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    ax, ay, az, aw = a
    bx, by, bz, bw = b
    return np.array(
        [
            aw * bx + ax * bw + ay * bz - az * by,
            aw * by + ay * bw + az * bx - ax * bz,
            aw * bz + az * bw + ax * by - ay * bx,
            aw * bw - ax * bx - ay * by - az * bz,
        ],
        dtype=np.float32,
    )


def _read_track(
    data: memoryview, start: int, end: int, width: int, rotation: bool = False
) -> Track:
    if end - start < 14:
        raise FormatError("truncated track")
    flags = struct.unpack_from("<H", data, start)[0]
    count = struct.unpack_from("<i", data, start + 10)[0]
    if count < 0:
        raise FormatError("negative track key count")
    cursor = start + 14
    keys: list[Key] = []
    cumulative = np.array([0, 0, 0, 1], dtype=np.float32)
    for _ in range(count):
        if cursor + 6 > end:
            raise FormatError("truncated track key")
        frame, key_flags = struct.unpack_from("<iH", data, cursor)
        cursor += 6
        parameters = [0.0] * 5
        for bit in range(5):
            if key_flags & (1 << bit):
                if cursor + 4 > end:
                    raise FormatError("truncated TCB parameters")
                parameters[bit] = struct.unpack_from("<f", data, cursor)[0]
                cursor += 4
        value_width = 4 if rotation else width
        if cursor + 4 * value_width > end:
            raise FormatError("truncated track value")
        value = np.array(
            struct.unpack_from(f"<{value_width}f", data, cursor), dtype=np.float32
        )
        cursor += 4 * value_width
        if rotation:
            delta = _axis_angle_quaternion(float(value[0]), value[1:])
            cumulative = _quat_multiply(delta, cumulative)
            value = cumulative.copy()
        keys.append(Key(frame, value, *parameters))
    return Track(keys, flags)


def _read_node(
    data: memoryview, start: int, end: int, node_type: int, default_id: int
) -> Node:
    node = Node("", node_type, default_id)
    for chunk_id, payload, chunk_end in _chunks(data, start, end):
        if chunk_id == NODE_ID:
            node.node_id = struct.unpack_from("<H", data, payload)[0]
        elif chunk_id == NODE_HDR:
            node.name, cursor = _read_string(data, payload, chunk_end)
            if cursor + 6 > chunk_end:
                raise FormatError("truncated node header")
            node.flags1, node.flags2, node.parent_id = struct.unpack_from(
                "<3H", data, cursor
            )
        elif chunk_id == PIVOT and node_type == 2:
            node.pivot = _vector(data, payload)
        elif chunk_id == INSTANCE_NAME and node_type == 2:
            node.instance_name, _ = _read_string(data, payload, chunk_end)
        elif chunk_id == POS_TRACK:
            node.position_track = _read_track(data, payload, chunk_end, 3)
        elif chunk_id == ROT_TRACK and node_type == 2:
            node.rotation_track = _read_track(
                data, payload, chunk_end, 4, rotation=True
            )
        elif chunk_id == SCL_TRACK and node_type == 2:
            node.scale_track = _read_track(data, payload, chunk_end, 3)
    return node


def loads(source: bytes | bytearray | memoryview) -> Scene:
    try:
        data = memoryview(source).cast("B")
        if len(data) < 6:
            raise FormatError("not a 3DS file: truncated root chunk")
        magic, size = struct.unpack_from("<HI", data, 0)
        if magic != M3DMAGIC:
            raise FormatError(f"not a 3DS file: root chunk is 0x{magic:04X}")
        if size < 6 or size > len(data):
            raise FormatError("invalid root chunk size")
        scene = Scene()
        for chunk_id, payload, chunk_end in _chunks(data, 6, size):
            if chunk_id == MDATA:
                for child, child_payload, child_end in _chunks(data, payload, chunk_end):
                    if child == MESH_VERSION:
                        scene.mesh_version = struct.unpack_from(
                            "<i", data, child_payload
                        )[0]
                    elif child == MASTER_SCALE:
                        scene.master_scale = struct.unpack_from(
                            "<f", data, child_payload
                        )[0]
                    elif child == AMBIENT_LIGHT:
                        scene.ambient = _read_color(data, child_payload, child_end)
                    elif child == MAT_ENTRY:
                        scene.materials.append(
                            _read_material(data, child_payload, child_end)
                        )
                    elif child == NAMED_OBJECT:
                        _read_named_object(scene, data, child_payload, child_end)
            elif chunk_id == KFDATA:
                node_number = 0
                for child, child_payload, child_end in _chunks(data, payload, chunk_end):
                    if child == KFHDR:
                        if child_end - child_payload < 2:
                            raise FormatError("truncated keyframe header")
                        scene.name, cursor = _read_string(
                            data, child_payload + 2, child_end
                        )
                        if cursor + 4 > child_end:
                            raise FormatError("truncated keyframe header")
                        scene.frames = struct.unpack_from("<i", data, cursor)[0]
                    elif child == KFSEG:
                        scene.segment_from, scene.segment_to = struct.unpack_from(
                            "<2i", data, child_payload
                        )
                    elif child == KFCURTIME:
                        scene.current_frame = struct.unpack_from(
                            "<i", data, child_payload
                        )[0]
                    elif child in NODE_TAGS:
                        scene.nodes.append(
                            _read_node(
                                data,
                                child_payload,
                                child_end,
                                NODE_TAGS[child],
                                node_number,
                            )
                        )
                        node_number += 1
        return scene
    except FormatError:
        raise
    except (BufferError, IndexError, TypeError, ValueError, struct.error) as error:
        raise FormatError(f"malformed 3DS payload: {error}") from error


def _write_color(value: np.ndarray) -> bytes:
    channels = bytes(
        max(0, min(255, math.floor(255.0 * float(channel) + 0.5)))
        for channel in value
    )
    return _chunk(COLOR_24, channels) + _chunk(LIN_COLOR_24, channels)


def _write_percentage(value: float) -> bytes:
    rounded = max(-128, min(127, math.floor(100.0 * value + 0.5)))
    return _chunk(INT_PERCENTAGE, struct.pack("<h", rounded))


def _write_texture(texture: TextureMap) -> bytes:
    if not texture.name:
        return b""
    payload = _write_percentage(texture.percent)
    payload += _chunk(MAT_MAPNAME, _cstring(texture.name))
    payload += _chunk(MAT_MAP_TILING, struct.pack("<H", texture.flags))
    payload += _chunk(MAT_MAP_TEXBLUR, struct.pack("<f", texture.blur))
    payload += _chunk(MAT_MAP_USCALE, struct.pack("<f", texture.scale[0]))
    payload += _chunk(MAT_MAP_VSCALE, struct.pack("<f", texture.scale[1]))
    payload += _chunk(MAT_MAP_UOFFSET, struct.pack("<f", texture.offset[0]))
    payload += _chunk(MAT_MAP_VOFFSET, struct.pack("<f", texture.offset[1]))
    payload += _chunk(MAT_MAP_ANG, struct.pack("<f", texture.rotation))
    return _chunk(MAT_TEXMAP, payload)


def _write_material(material: Material) -> bytes:
    payload = _chunk(MAT_NAME, _cstring(material.name))
    payload += _chunk(MAT_AMBIENT, _write_color(material.ambient))
    payload += _chunk(MAT_DIFFUSE, _write_color(material.diffuse))
    payload += _chunk(MAT_SPECULAR, _write_color(material.specular))
    payload += _chunk(MAT_SHININESS, _write_percentage(material.shininess))
    payload += _chunk(MAT_SHIN2PCT, _write_percentage(material.shin_strength))
    payload += _chunk(
        MAT_TRANSPARENCY, _write_percentage(material.transparency)
    )
    payload += _chunk(MAT_SHADING, struct.pack("<h", material.shading))
    if material.self_illum:
        payload += _chunk(MAT_SELF_ILLUM)
    if material.two_sided:
        payload += _chunk(MAT_TWO_SIDE)
    if material.use_wire:
        payload += _chunk(MAT_WIRE)
    payload += _chunk(MAT_WIRE_SIZE, struct.pack("<f", material.wire_size))
    payload += _write_texture(material.texture1_map)
    return _chunk(MAT_ENTRY, payload)


def _write_mesh(mesh: Mesh) -> bytes:
    if len(mesh.vertices) > 65535 or len(mesh.faces) > 65535:
        raise ValueError("3DS stores mesh point and face counts as uint16")
    if mesh.texcoords is not None and len(mesh.texcoords) > 65535:
        raise ValueError("3DS stores texture coordinate counts as uint16")
    vertices = mesh.vertices
    if np.linalg.det(mesh.matrix) < 0.0 and len(vertices):
        inverse = np.linalg.inv(mesh.matrix)
        scaled = mesh.matrix.copy()
        scaled[0, :] *= -1.0
        correction = inverse @ scaled
        vertices = mesh.transformed_vertices(
            np.ascontiguousarray(correction, dtype=np.float32)
        )
    payload = b""
    if len(vertices):
        payload += _chunk(
            POINT_ARRAY,
            struct.pack("<H", len(vertices))
            + np.asarray(vertices, dtype="<f4").tobytes(),
        )
    if mesh.texcoords is not None and len(mesh.texcoords):
        payload += _chunk(
            TEX_VERTS,
            struct.pack("<H", len(mesh.texcoords))
            + np.asarray(mesh.texcoords, dtype="<f4").tobytes(),
        )
    matrix_values = [
        float(mesh.matrix[column, row])
        for column in range(4)
        for row in range(3)
    ]
    payload += _chunk(MESH_MATRIX, struct.pack("<12f", *matrix_values))
    if mesh.color:
        payload += _chunk(MESH_COLOR, bytes([mesh.color]))
    if len(mesh.faces):
        face_payload = struct.pack("<H", len(mesh.faces))
        records = np.column_stack((mesh.faces, mesh.face_flags)).astype("<u2")
        face_payload += records.tobytes()
        seen: set[str] = set()
        for material in mesh.face_materials:
            if not material or material in seen:
                continue
            seen.add(material)
            indices = [
                index
                for index, value in enumerate(mesh.face_materials)
                if value == material
            ]
            face_payload += _chunk(
                MSH_MAT_GROUP,
                _cstring(material)
                + struct.pack("<H", len(indices))
                + struct.pack(f"<{len(indices)}H", *indices),
            )
        face_payload += _chunk(
            SMOOTH_GROUP, np.asarray(mesh.smoothing, dtype="<u4").tobytes()
        )
        payload += _chunk(FACE_ARRAY, face_payload)
    return _chunk(N_TRI_OBJECT, payload)


def _object_flag_chunks(flags: int) -> bytes:
    mapping = [
        (0x01, OBJ_HIDDEN),
        (0x02, OBJ_VIS_LOFTER),
        (0x04, OBJ_DOESNT_CAST),
        (0x08, OBJ_MATTE),
        (0x10, OBJ_DONT_RCVSHADOW),
        (0x20, OBJ_FAST),
        (0x40, OBJ_FROZEN),
    ]
    return b"".join(_chunk(chunk_id) for bit, chunk_id in mapping if flags & bit)


def _write_camera(camera: Camera) -> bytes:
    lens = 2400.0 / (camera.fov if abs(camera.fov) >= 1.0e-8 else 45.0)
    payload = struct.pack(
        "<8f", *camera.position, *camera.target, camera.roll, lens
    )
    if camera.see_cone:
        payload += _chunk(CAM_SEE_CONE)
    payload += _chunk(
        CAM_RANGES, struct.pack("<2f", camera.near_range, camera.far_range)
    )
    return _chunk(N_CAMERA, payload)


def _write_light(light: Light) -> bytes:
    payload = struct.pack("<3f", *light.position)
    payload += _chunk(COLOR_F, struct.pack("<3f", *light.color))
    if light.off:
        payload += _chunk(DL_OFF)
    payload += _chunk(DL_OUTER_RANGE, struct.pack("<f", light.outer_range))
    payload += _chunk(DL_INNER_RANGE, struct.pack("<f", light.inner_range))
    payload += _chunk(DL_MULTIPLIER, struct.pack("<f", light.multiplier))
    if light.attenuation:
        payload += _chunk(DL_ATTENUATE)
    if light.spot_light:
        spot = struct.pack(
            "<5f", *light.spot, light.hot_spot, light.fall_off
        )
        spot += _chunk(DL_SPOT_ROLL, struct.pack("<f", light.roll))
        if light.shadowed:
            spot += _chunk(DL_SHADOWED)
        if light.see_cone:
            spot += _chunk(DL_SEE_CONE)
        if light.rectangular_spot:
            spot += _chunk(DL_SPOT_RECTANGULAR)
        payload += _chunk(DL_SPOTLIGHT, spot)
    return _chunk(N_DIRECT_LIGHT, payload)


def _write_key_prefix(key: Key) -> bytes:
    values = [
        key.tension,
        key.continuity,
        key.bias,
        key.ease_to,
        key.ease_from,
    ]
    flags = sum((1 << index) for index, value in enumerate(values) if value)
    payload = struct.pack("<iH", key.frame, flags)
    payload += b"".join(
        struct.pack("<f", value) for index, value in enumerate(values) if flags & (1 << index)
    )
    return payload


def _quat_inverse(quaternion: np.ndarray) -> np.ndarray:
    result = quaternion.copy()
    result[:3] *= -1
    norm = float(np.dot(result, result))
    return result / norm if norm else np.array([0, 0, 0, 1], dtype=np.float32)


def _quat_to_axis_angle(quaternion: np.ndarray) -> tuple[float, np.ndarray]:
    q = quaternion / max(float(np.linalg.norm(quaternion)), 1.0e-30)
    length = float(np.linalg.norm(q[:3]))
    if length < 1.0e-8:
        return 0.0, np.array([1, 0, 0], dtype=np.float32)
    angle = -2.0 * math.atan2(length, float(q[3]))
    axis = q[:3] / math.sin(-0.5 * angle)
    return angle, axis


def _write_track(track: Track, rotation: bool = False) -> bytes:
    keys = sorted(track.keys, key=lambda key: key.frame)
    payload = struct.pack("<HIIi", track.flags, 0, 0, len(keys))
    previous = np.array([0, 0, 0, 1], dtype=np.float32)
    for key in keys:
        payload += _write_key_prefix(key)
        if rotation:
            delta = _quat_multiply(key.value, _quat_inverse(previous))
            angle, axis = _quat_to_axis_angle(delta)
            payload += struct.pack("<4f", angle, *axis)
            previous = key.value
        else:
            payload += np.asarray(key.value, dtype="<f4").tobytes()
    return payload


def _write_node(node: Node) -> bytes:
    payload = _chunk(NODE_ID, struct.pack("<H", node.node_id))
    payload += _chunk(
        NODE_HDR,
        _cstring(node.name)
        + struct.pack("<3H", node.flags1, node.flags2, node.parent_id),
    )
    if node.type == 2:
        payload += _chunk(PIVOT, struct.pack("<3f", *node.pivot))
        if node.instance_name:
            payload += _chunk(INSTANCE_NAME, _cstring(node.instance_name))
    if node.position_track.keys:
        payload += _chunk(POS_TRACK, _write_track(node.position_track))
    if node.type == 2 and node.rotation_track.keys:
        payload += _chunk(
            ROT_TRACK, _write_track(node.rotation_track, rotation=True)
        )
    if node.type == 2 and node.scale_track.keys:
        payload += _chunk(SCL_TRACK, _write_track(node.scale_track))
    return _chunk(TYPE_TAGS.get(node.type, 0xB002), payload)


def dumps(scene: Scene) -> bytes:
    mdata = _chunk(MESH_VERSION, struct.pack("<i", scene.mesh_version))
    mdata += _chunk(MASTER_SCALE, struct.pack("<f", scene.master_scale))
    mdata += _chunk(
        AMBIENT_LIGHT, _chunk(COLOR_F, struct.pack("<3f", *scene.ambient))
    )
    mdata += b"".join(_write_material(material) for material in scene.materials)
    for mesh in scene.meshes:
        mdata += _chunk(
            NAMED_OBJECT,
            _cstring(mesh.name)
            + _write_mesh(mesh)
            + _object_flag_chunks(mesh.object_flags),
        )
    for camera in scene.cameras:
        mdata += _chunk(
            NAMED_OBJECT,
            _cstring(camera.name)
            + _write_camera(camera)
            + _object_flag_chunks(camera.object_flags),
        )
    for light in scene.lights:
        mdata += _chunk(
            NAMED_OBJECT,
            _cstring(light.name)
            + _write_light(light)
            + _object_flag_chunks(light.object_flags),
        )
    payload = _chunk(M3D_VERSION, struct.pack("<I", 3)) + _chunk(MDATA, mdata)
    if scene.nodes:
        keyframes = _chunk(
            KFHDR,
            struct.pack("<H", 5)
            + _cstring(scene.name[:12])
            + struct.pack("<i", scene.frames),
        )
        keyframes += _chunk(
            KFSEG, struct.pack("<2i", scene.segment_from, scene.segment_to)
        )
        keyframes += _chunk(KFCURTIME, struct.pack("<i", scene.current_frame))
        keyframes += b"".join(_write_node(node) for node in scene.nodes)
        payload += _chunk(KFDATA, keyframes)
    return _chunk(M3DMAGIC, payload)
