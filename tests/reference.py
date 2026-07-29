from __future__ import annotations

import math

import numpy as np


# lib3ds: lib3ds/vector.c lib3ds_vector_normalize
def normalize(vector: np.ndarray) -> np.ndarray:
    vector = np.asarray(vector, dtype=np.float32).copy()
    length = np.float32(math.sqrt(float(np.dot(vector, vector))))
    if abs(length) < 1.0e-8:
        if vector[0] >= vector[1] and vector[0] >= vector[2]:
            return np.array([1, 0, 0], dtype=np.float32)
        if vector[1] >= vector[2]:
            return np.array([0, 1, 0], dtype=np.float32)
        return np.array([0, 0, 1], dtype=np.float32)
    return np.asarray(vector * np.float32(1.0 / length), dtype=np.float32)


# lib3ds: lib3ds/mesh.c lib3ds_mesh_calculate_normals
def normals(
    vertices: np.ndarray, faces: np.ndarray, smoothing: np.ndarray
) -> np.ndarray:
    face_normals = []
    incident: list[list[int]] = [[] for _ in vertices]
    for index, face in enumerate(faces):
        a, b, c = vertices[face]
        face_normals.append(normalize(np.cross(c - b, a - b)))
        for vertex in face:
            incident[vertex].insert(0, index)
    face_normals = np.asarray(face_normals, dtype=np.float32)
    result = np.empty((len(faces), 3, 3), dtype=np.float32)
    for face_index, face in enumerate(faces):
        for corner, vertex in enumerate(face):
            if smoothing[face_index]:
                value = np.zeros(3, dtype=np.float32)
                seen: list[np.ndarray] = []
                for other in incident[vertex]:
                    normal = face_normals[other]
                    duplicate = any(
                        abs(float(np.dot(item, normal)) - 1.0) < 1.0e-5
                        for item in seen
                    )
                    if not duplicate and smoothing[face_index] & smoothing[other]:
                        value = np.asarray(value + normal, dtype=np.float32)
                        seen.append(normal)
            else:
                value = face_normals[face_index]
            result[face_index, corner] = normalize(value)
    return result


# lib3ds: lib3ds/ease.c lib3ds_ease
def ease(
    previous: float,
    current: float,
    following: float,
    ease_from: float,
    ease_to: float,
) -> float:
    step = (current - previous) / (following - previous)
    value = step
    total = ease_to + ease_from
    if total != 0:
        if total > 1:
            ease_to /= total
            ease_from /= total
        coefficient = 1 / (2 - (ease_to + ease_from))
        if step < ease_from:
            value = coefficient / ease_from * step * step
        elif 1 - ease_to <= step:
            step = 1 - step
            value = 1 - coefficient / ease_to * step * step
        else:
            value = (2 * step - ease_from) * coefficient
    return value
