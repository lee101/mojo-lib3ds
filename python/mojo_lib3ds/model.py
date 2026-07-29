from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import ClassVar

import numpy as np
from numpy.typing import NDArray

from ._lib import address, lib

F32 = NDArray[np.float32]
I64 = NDArray[np.int64]
U32 = NDArray[np.uint32]

_TRANSFORM_PARALLEL_THRESHOLD = 262_144
_TRANSFORM_WORKERS = 16
_transform_pool = ThreadPoolExecutor(max_workers=_TRANSFORM_WORKERS)


def _f32(value, shape: tuple[int, ...] | None = None) -> F32:
    source = np.asarray(value)
    if source.dtype.kind not in "biuf":
        raise TypeError("expected a real numeric array")
    if source.size and (
        not np.isfinite(source).all()
        or np.any(np.abs(source.astype(np.float64, copy=False)) > np.finfo(np.float32).max)
    ):
        raise ValueError("value cannot be represented as finite float32")
    array = np.ascontiguousarray(source, dtype=np.float32)
    if shape is not None and array.shape != shape:
        raise ValueError(f"expected shape {shape}, got {array.shape}")
    return array


def _integer_array(value, dtype: np.dtype, name: str) -> np.ndarray:
    source = np.asarray(value)
    if source.dtype.kind not in "biu":
        raise TypeError(f"{name} must contain integers")
    info = np.iinfo(dtype)
    if source.size and (
        np.any(source < info.min) or np.any(source > info.max)
    ):
        raise ValueError(f"{name} value is outside {np.dtype(dtype).name} range")
    return np.ascontiguousarray(source, dtype=dtype)


@dataclass(slots=True)
class TextureMap:
    name: str = ""
    flags: int = 0x10
    percent: float = 1.0
    blur: float = 0.0
    scale: F32 = field(default_factory=lambda: np.ones(2, dtype=np.float32))
    offset: F32 = field(default_factory=lambda: np.zeros(2, dtype=np.float32))
    rotation: float = 0.0


@dataclass(slots=True)
class Material:
    name: str = ""
    ambient: F32 = field(
        default_factory=lambda: np.full(3, np.float32(0.588235))
    )
    diffuse: F32 = field(
        default_factory=lambda: np.full(3, np.float32(0.588235))
    )
    specular: F32 = field(
        default_factory=lambda: np.full(3, np.float32(0.898039))
    )
    shininess: float = 0.1
    shin_strength: float = 0.0
    transparency: float = 0.0
    shading: int = 3
    two_sided: bool = False
    self_illum: bool = False
    use_wire: bool = False
    wire_size: float = 1.0
    texture1_map: TextureMap = field(default_factory=TextureMap)


@dataclass(slots=True)
class Mesh:
    name: str
    vertices: F32
    faces: I64
    smoothing: U32 | None = None
    face_flags: NDArray[np.uint16] | None = None
    face_materials: list[str] | None = None
    texcoords: F32 | None = None
    matrix: F32 = field(
        default_factory=lambda: np.ascontiguousarray(np.eye(4), dtype=np.float32)
    )
    object_flags: int = 0
    color: int = 0

    def __post_init__(self) -> None:
        self.vertices = _f32(self.vertices)
        self.faces = _integer_array(self.faces, np.dtype(np.int64), "faces")
        if self.vertices.ndim != 2 or self.vertices.shape[1:] != (3,):
            raise ValueError("vertices must have shape (n, 3)")
        if self.faces.ndim != 2 or self.faces.shape[1:] != (3,):
            raise ValueError("faces must have shape (m, 3)")
        if self.faces.size and (
            int(self.faces.min()) < 0
            or int(self.faces.max()) >= len(self.vertices)
        ):
            raise ValueError("face vertex index out of range")
        count = len(self.faces)
        self.smoothing = (
            np.zeros(count, dtype=np.uint32)
            if self.smoothing is None
            else _integer_array(self.smoothing, np.dtype(np.uint32), "smoothing")
        )
        self.face_flags = (
            np.full(count, 7, dtype=np.uint16)
            if self.face_flags is None
            else _integer_array(
                self.face_flags, np.dtype(np.uint16), "face_flags"
            )
        )
        if self.smoothing.shape != (count,) or self.face_flags.shape != (count,):
            raise ValueError("smoothing and face_flags must have one value per face")
        if self.face_materials is None:
            self.face_materials = [""] * count
        if len(self.face_materials) != count:
            raise ValueError("face_materials must have one value per face")
        if self.texcoords is not None:
            self.texcoords = _f32(self.texcoords)
            if self.texcoords.ndim != 2 or self.texcoords.shape[1] != 2:
                raise ValueError("texcoords must have shape (n, 2)")
            if len(self.texcoords) > 65535:
                raise ValueError("3DS stores texture coordinate counts as uint16")
        self.matrix = _f32(self.matrix, (4, 4))

    def calculate_normals(self) -> F32:
        """Return lib3ds corner normals with shape ``(faces, 3, 3)``."""
        face_count = len(self.faces)
        result = np.empty((face_count, 3, 3), dtype=np.float32)
        if face_count == 0:
            return result
        counts = np.bincount(self.faces.ravel(), minlength=len(self.vertices))
        offsets = np.empty(len(self.vertices) + 1, dtype=np.int64)
        offsets[0] = 0
        np.cumsum(counts, out=offsets[1:])
        adjacent = np.empty(3 * face_count, dtype=np.int64)
        cursor = offsets[1:].copy()
        # lib3ds prepends each incident face, so each vertex list is descending.
        for face in range(face_count):
            for vertex in self.faces[face]:
                cursor[vertex] -= 1
                adjacent[cursor[vertex]] = face
        face_normals = np.empty((face_count, 3), dtype=np.float32)
        lib.m3ds_mesh_normals(
            address(self.vertices),
            address(self.faces),
            address(self.smoothing),
            address(offsets),
            address(adjacent),
            address(face_normals),
            address(result),
            face_count,
        )
        return result

    def bounding_box(self) -> tuple[F32, F32]:
        result = np.empty(6, dtype=np.float32)
        if len(self.vertices):
            vertex_address = address(self.vertices)
        else:
            vertex_address = address(np.empty(1, dtype=np.float32))
        lib.m3ds_mesh_bounding_box(vertex_address, len(self.vertices), address(result))
        return result[:3].copy(), result[3:].copy()

    def transformed_vertices(self, matrix: F32 | None = None) -> F32:
        transform = self.matrix if matrix is None else _f32(matrix, (4, 4))
        result = np.empty_like(self.vertices)
        point_count = len(self.vertices)
        if point_count < _TRANSFORM_PARALLEL_THRESHOLD:
            if point_count:
                lib.m3ds_transform_points(
                    address(self.vertices),
                    address(transform),
                    address(result),
                    point_count,
                )
        else:
            chunk_size = (point_count + _TRANSFORM_WORKERS - 1) // _TRANSFORM_WORKERS
            futures = []
            for start in range(0, point_count, chunk_size):
                count = min(chunk_size, point_count - start)
                futures.append(
                    _transform_pool.submit(
                        lib.m3ds_transform_points,
                        address(self.vertices[start:]),
                        address(transform),
                        address(result[start:]),
                        count,
                    )
                )
            for future in futures:
                future.result()
        return result


@dataclass(slots=True)
class Camera:
    name: str
    position: F32 = field(default_factory=lambda: np.zeros(3, dtype=np.float32))
    target: F32 = field(default_factory=lambda: np.zeros(3, dtype=np.float32))
    roll: float = 0.0
    fov: float = 45.0
    see_cone: bool = False
    near_range: float = 0.0
    far_range: float = 0.0
    object_flags: int = 0


@dataclass(slots=True)
class Light:
    name: str
    position: F32 = field(default_factory=lambda: np.zeros(3, dtype=np.float32))
    color: F32 = field(default_factory=lambda: np.zeros(3, dtype=np.float32))
    off: bool = False
    outer_range: float = 0.0
    inner_range: float = 0.0
    multiplier: float = 0.0
    attenuation: float = 0.0
    spot_light: bool = False
    spot: F32 = field(default_factory=lambda: np.zeros(3, dtype=np.float32))
    roll: float = 0.0
    hot_spot: float = 0.0
    fall_off: float = 0.0
    see_cone: bool = False
    rectangular_spot: bool = False
    shadowed: bool = False
    object_flags: int = 0


@dataclass(slots=True)
class Key:
    frame: int
    value: F32
    tension: float = 0.0
    continuity: float = 0.0
    bias: float = 0.0
    ease_to: float = 0.0
    ease_from: float = 0.0


@dataclass(slots=True)
class Track:
    keys: list[Key] = field(default_factory=list)
    flags: int = 0

    REPEAT: ClassVar[int] = 0x0001

    def evaluate(self, time: float, default: F32) -> F32:
        if not self.keys:
            return np.array(default, dtype=np.float32, copy=True)
        keys = sorted(self.keys, key=lambda key: key.frame)
        if len(keys) == 1:
            return keys[0].value.copy()
        if time < keys[0].frame and self.flags & self.REPEAT:
            return keys[0].value.copy()
        if time >= keys[-1].frame:
            if not self.flags & self.REPEAT:
                return keys[-1].value.copy()
            span = keys[-1].frame - keys[0].frame
            time = (time - keys[0].frame) % span + keys[0].frame
        right = next(
            (index for index in range(1, len(keys)) if time < keys[index].frame),
            len(keys) - 1,
        )
        left = right - 1
        u = (time - keys[left].frame) / (keys[right].frame - keys[left].frame)
        if keys[left].value.shape == (4,):
            value = keys[left].value * (1.0 - u) + keys[right].value * u
            length = float(np.linalg.norm(value))
            return np.ascontiguousarray(
                value / length if length >= 1.0e-8 else [0, 0, 0, 1],
                dtype=np.float32,
            )

        # lib3ds: lib3ds/tracks.c lib3ds_lin3_track_setup/eval
        incoming = [np.zeros_like(key.value) for key in keys]
        outgoing = [np.zeros_like(key.value) for key in keys]
        incoming[0] = outgoing[0] = keys[1].value - keys[0].value
        incoming[-1] = outgoing[-1] = keys[-1].value - keys[-2].value
        for index in range(1, len(keys) - 1):
            previous, current, following = (
                keys[index - 1],
                keys[index],
                keys[index + 1],
            )
            duration = np.float32(
                0.5 * (current.frame - previous.frame + following.frame - current.frame)
            )
            fp = np.float32((current.frame - previous.frame) / duration)
            fn = np.float32((following.frame - current.frame) / duration)
            continuity_abs = np.float32(abs(current.continuity))
            fp = fp + continuity_abs - continuity_abs * fp
            fn = fn + continuity_abs - continuity_abs * fn
            cm = np.float32(1.0 - current.continuity)
            tm = np.float32(0.5 * (1.0 - current.tension))
            cp = np.float32(2.0 - cm)
            bm = np.float32(1.0 - current.bias)
            bp = np.float32(2.0 - bm)
            previous_delta = current.value - previous.value
            following_delta = following.value - current.value
            outgoing[index] = (
                tm * cm * bp * fp * previous_delta
                + tm * cp * bm * fp * following_delta
            )
            incoming[index] = (
                tm * cp * bp * fn * previous_delta
                + tm * cm * bm * fn * following_delta
            )
        u = np.float32(u)
        x = np.float32(2 * u**3 - 3 * u**2 + 1)
        y = np.float32(-2 * u**3 + 3 * u**2)
        z = np.float32(u**3 - 2 * u**2 + u)
        w = np.float32(u**3 - u**2)
        return np.ascontiguousarray(
            x * keys[left].value
            + y * keys[right].value
            + z * incoming[left]
            + w * outgoing[right],
            dtype=np.float32,
        )


@dataclass(slots=True)
class Node:
    name: str
    type: int = 2
    node_id: int = 0
    parent_id: int = 65535
    flags1: int = 0
    flags2: int = 0
    pivot: F32 = field(default_factory=lambda: np.zeros(3, dtype=np.float32))
    instance_name: str = ""
    position_track: Track = field(default_factory=Track)
    rotation_track: Track = field(default_factory=Track)
    scale_track: Track = field(default_factory=Track)
    matrix: F32 = field(
        default_factory=lambda: np.ascontiguousarray(np.eye(4), dtype=np.float32)
    )


@dataclass(slots=True)
class Scene:
    meshes: list[Mesh] = field(default_factory=list)
    materials: list[Material] = field(default_factory=list)
    cameras: list[Camera] = field(default_factory=list)
    lights: list[Light] = field(default_factory=list)
    nodes: list[Node] = field(default_factory=list)
    ambient: F32 = field(default_factory=lambda: np.zeros(3, dtype=np.float32))
    master_scale: float = 1.0
    mesh_version: int = 3
    name: str = "MOJO3DS"
    frames: int = 100
    segment_from: int = 0
    segment_to: int = 100
    current_frame: int = 0

    @classmethod
    def from_bytes(cls, data: bytes | bytearray | memoryview) -> "Scene":
        from .io import loads

        return loads(data)

    @classmethod
    def load(cls, path: str | Path) -> "Scene":
        return cls.from_bytes(Path(path).read_bytes())

    def to_bytes(self) -> bytes:
        from .io import dumps

        return dumps(self)

    def save(self, path: str | Path) -> None:
        Path(path).write_bytes(self.to_bytes())

    def evaluate(self, time: float) -> list[F32]:
        if not self.nodes:
            return []
        by_id = {node.node_id: index for index, node in enumerate(self.nodes)}
        remaining = set(range(len(self.nodes)))
        order: list[int] = []
        while remaining:
            progressed = False
            for index in list(remaining):
                parent_id = self.nodes[index].parent_id
                if parent_id == 65535 or by_id.get(parent_id) in order:
                    order.append(index)
                    remaining.remove(index)
                    progressed = True
            if not progressed:
                raise ValueError("node hierarchy contains a cycle or missing parent")
        ordered = [self.nodes[index] for index in order]
        ordered_index = {node.node_id: index for index, node in enumerate(ordered)}
        parents = np.array(
            [
                -1
                if node.parent_id == 65535
                else ordered_index[node.parent_id]
                for node in ordered
            ],
            dtype=np.int64,
        )
        types = np.array([node.type for node in ordered], dtype=np.int64)
        positions = np.stack(
            [
                node.position_track.evaluate(time, np.zeros(3, dtype=np.float32))
                for node in ordered
            ]
        ).astype(np.float32)
        rotations = np.stack(
            [
                node.rotation_track.evaluate(
                    time, np.array([0, 0, 0, 1], dtype=np.float32)
                )
                for node in ordered
            ]
        ).astype(np.float32)
        scales = np.stack(
            [
                node.scale_track.evaluate(time, np.ones(3, dtype=np.float32))
                for node in ordered
            ]
        ).astype(np.float32)
        matrices = np.empty((len(ordered), 4, 4), dtype=np.float32)
        ok = lib.m3ds_compose_nodes(
            address(parents),
            address(types),
            address(positions),
            address(rotations),
            address(scales),
            address(matrices),
            len(ordered),
        )
        if not ok:
            raise ValueError("parents must precede children")
        result: list[F32] = [np.empty((4, 4), dtype=np.float32)] * len(self.nodes)
        for ordered_slot, original_slot in enumerate(order):
            matrix = matrices[ordered_slot].copy()
            self.nodes[original_slot].matrix = matrix
            result[original_slot] = matrix
        return result
