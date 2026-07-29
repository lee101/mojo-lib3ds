from __future__ import annotations

import numpy as np
import pytest

from mojo_lib3ds import Mesh
from mojo_lib3ds._lib import lib
from mojo_lib3ds.model import _TRANSFORM_PARALLEL_THRESHOLD

from reference import ease, normals


def mesh(vertices, faces, smoothing=None) -> Mesh:
    return Mesh(
        "test",
        np.asarray(vertices, dtype=np.float32).reshape((-1, 3)),
        np.asarray(faces, dtype=np.int64).reshape((-1, 3)),
        None if smoothing is None else np.asarray(smoothing, dtype=np.uint32),
    )


@pytest.mark.parametrize(
    ("vertices", "faces", "smoothing"),
    [
        ([], [], []),
        ([[0, 0, 0], [1, 0, 0], [0, 1, 0]], [[0, 1, 2]], [0]),
        (
            [[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]],
            [[0, 1, 2], [0, 2, 3]],
            [1, 1],
        ),
        (
            [[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]],
            [[0, 1, 2], [0, 3, 1]],
            [1, 1],
        ),
        (
            [[0, 0, 0], [0, 0, 0], [0, 0, 0]],
            [[0, 1, 2]],
            [0],
        ),
        (
            [[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, -1, 0], [0, 0, 1]],
            [[0, 1, 2], [0, 3, 1], [0, 1, 4]],
            [1, 1, 2],
        ),
        (
            [[0, 0, 0], [1, 0, 0], [0, 1, 0], [99, 99, 99]],
            [[0, 1, 2], [0, 1, 2]],
            [1, 1],
        ),
    ],
)
def test_normals_match_upstream_translation(vertices, faces, smoothing):
    subject = mesh(vertices, faces, smoothing)
    expected = normals(subject.vertices, subject.faces, subject.smoothing)
    np.testing.assert_allclose(subject.calculate_normals(), expected, atol=2e-7)


def test_invalid_and_unreferenced_vertices():
    subject = mesh(
        [[0, 0, 0], [1, 0, 0], [0, 1, 0], [7, 8, 9]],
        [[0, 1, 2]],
    )
    assert subject.calculate_normals().shape == (1, 3, 3)
    with pytest.raises(ValueError, match="out of range"):
        mesh([[0, 0, 0]], [[0, 1, 0]])


def test_high_valence_normals_have_no_fixed_adjacency_limit():
    angles = np.linspace(0, 2 * np.pi, 260, endpoint=False, dtype=np.float32)
    vertices = np.vstack(
        (
            np.zeros((1, 3), dtype=np.float32),
            np.column_stack((np.cos(angles), np.sin(angles), np.zeros_like(angles))),
        )
    )
    faces = np.column_stack(
        (
            np.zeros(len(angles), dtype=np.int64),
            np.arange(1, len(angles) + 1, dtype=np.int64),
            np.roll(np.arange(1, len(angles) + 1, dtype=np.int64), -1),
        )
    )
    subject = mesh(vertices, faces, np.ones(len(faces), dtype=np.uint32))
    np.testing.assert_allclose(
        subject.calculate_normals(),
        normals(subject.vertices, subject.faces, subject.smoothing),
        atol=2e-7,
    )


@pytest.mark.parametrize(
    ("keyword", "value", "message"),
    [
        ("faces", np.array([[0, 2**63, 0]], dtype=np.uint64), "int64 range"),
        ("smoothing", [-1], "uint32 range"),
        ("face_flags", [65536], "uint16 range"),
    ],
)
def test_integer_inputs_do_not_silently_wrap(keyword, value, message):
    arguments = {
        "vertices": np.zeros((1, 3), dtype=np.float32),
        "faces": np.zeros((1, 3), dtype=np.int64),
        "smoothing": np.zeros(1, dtype=np.uint32),
        keyword: value,
    }
    with pytest.raises(ValueError, match=message):
        Mesh("bad", **arguments)


def test_float_inputs_reject_nonfinite_and_overflow():
    with pytest.raises(ValueError, match="finite float32"):
        Mesh("bad", [[np.inf, 0, 0]], np.empty((0, 3), dtype=np.int64))
    with pytest.raises(ValueError, match="finite float32"):
        Mesh(
            "bad",
            [[np.finfo(np.float64).max, 0, 0]],
            np.empty((0, 3), dtype=np.int64),
        )


def test_bounding_box_preserves_lib3ds_flt_min_edge_case():
    subject = mesh([[-3, -2, -1], [-1, -4, -2]], [])
    minimum, maximum = subject.bounding_box()
    np.testing.assert_array_equal(minimum, [-3, -4, -2])
    np.testing.assert_array_equal(
        maximum, np.full(3, np.finfo(np.float32).tiny, dtype=np.float32)
    )


def test_empty_bounding_box_matches_upstream_initializers():
    minimum, maximum = mesh([], []).bounding_box()
    np.testing.assert_array_equal(
        minimum, np.full(3, np.finfo(np.float32).max, dtype=np.float32)
    )
    np.testing.assert_array_equal(
        maximum, np.full(3, np.finfo(np.float32).tiny, dtype=np.float32)
    )


def test_transform_uses_lib3ds_column_major_layout():
    subject = mesh([[1, 2, 3], [-1, 0, 4]], [])
    matrix = np.array(
        [[2, 0, 0, 0], [0, 3, 0, 0], [0, 0, 4, 0], [5, 6, 7, 1]],
        dtype=np.float32,
    )
    expected = np.array([[7, 12, 19], [3, 6, 23]], dtype=np.float32)
    np.testing.assert_array_equal(subject.transformed_vertices(matrix), expected)


@pytest.mark.parametrize("point_count", [1, 2, 3, 4, 5, 7, 9])
def test_transform_simd_tail(point_count):
    vertices = np.arange(point_count * 3, dtype=np.float32).reshape((-1, 3))
    subject = mesh(vertices, [])
    matrix = np.array(
        [[2, 0, 0, 0], [0, 3, 0, 0], [0, 0, 4, 0], [5, 6, 7, 1]],
        dtype=np.float32,
    )
    expected = vertices * np.array([2, 3, 4], dtype=np.float32) + np.array(
        [5, 6, 7], dtype=np.float32
    )
    np.testing.assert_array_equal(subject.transformed_vertices(matrix), expected)


@pytest.mark.parametrize(
    "point_count",
    [_TRANSFORM_PARALLEL_THRESHOLD - 1, _TRANSFORM_PARALLEL_THRESHOLD + 3],
)
def test_transform_parallel_threshold(point_count):
    vertices = np.arange(point_count * 3, dtype=np.float32).reshape((-1, 3))
    subject = mesh(vertices, [])
    matrix = np.array(
        [[2, 0, 0, 0], [0, 3, 0, 0], [0, 0, 4, 0], [5, 6, 7, 1]],
        dtype=np.float32,
    )
    expected = vertices * np.array([2, 3, 4], dtype=np.float32) + np.array(
        [5, 6, 7], dtype=np.float32
    )
    np.testing.assert_array_equal(subject.transformed_vertices(matrix), expected)


@pytest.mark.parametrize(
    ("values", "parameters"),
    [
        ((0, 5, 10), (0.0, 0.0)),
        ((0, 1, 10), (0.4, 0.2)),
        ((0, 9, 10), (0.2, 0.4)),
        ((0, 5, 10), (0.8, 0.7)),
    ],
)
def test_ease_matches_upstream(values, parameters):
    expected = ease(*values, *parameters)
    actual = lib.m3ds_ease(
        *(np.float32(value) for value in (*values, *parameters))
    )
    assert actual == pytest.approx(expected, abs=2e-7)
