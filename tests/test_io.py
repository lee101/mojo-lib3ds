from __future__ import annotations

import struct

import numpy as np
import pytest
from pyassimp import load as assimp_load

from mojo_lib3ds import (
    Camera,
    Key,
    Light,
    Material,
    Mesh,
    Node,
    Scene,
    TextureMap,
    Track,
)
from mojo_lib3ds.io import FormatError


def sample_scene() -> Scene:
    material = Material(
        name="paint",
        ambient=np.array([0.1, 0.2, 0.3], dtype=np.float32),
        diffuse=np.array([0.7, 0.4, 0.2], dtype=np.float32),
        specular=np.array([1.0, 0.9, 0.8], dtype=np.float32),
        shininess=0.42,
        transparency=0.17,
        two_sided=True,
        texture1_map=TextureMap(
            name="grid.png",
            scale=np.array([2, 3], np.float32),
            offset=np.array([0.25, 0.5], np.float32),
        ),
    )
    mesh = Mesh(
        "quad",
        np.array(
            [[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0], [9, 9, 9]],
            dtype=np.float32,
        ),
        np.array([[0, 1, 2], [0, 2, 3]], dtype=np.int64),
        smoothing=np.array([1, 1], dtype=np.uint32),
        face_materials=["paint", "paint"],
        texcoords=np.array(
            [[0, 0], [1, 0], [1, 1], [0, 1], [0, 0]], dtype=np.float32
        ),
        object_flags=0x41,
    )
    camera = Camera(
        "camera",
        np.array([4, 5, 6], np.float32),
        np.array([0, 0, 0], np.float32),
        roll=0.2,
        fov=55,
        see_cone=True,
        near_range=0.1,
        far_range=100,
    )
    light = Light(
        "key",
        position=np.array([3, 2, 8], np.float32),
        color=np.array([1, 0.8, 0.6], np.float32),
        multiplier=1.5,
        spot_light=True,
        spot=np.array([0, 0, 0], np.float32),
        hot_spot=30,
        fall_off=45,
        shadowed=True,
    )
    parent = Node(
        "parent",
        node_id=10,
        position_track=Track(
            [
                Key(0, np.array([1, 0, 0], np.float32)),
                Key(10, np.array([3, 0, 0], np.float32)),
            ]
        ),
    )
    child = Node(
        "child",
        node_id=11,
        parent_id=10,
        position_track=Track([Key(0, np.array([0, 2, 0], np.float32))]),
        scale_track=Track([Key(0, np.ones(3, np.float32))]),
    )
    return Scene(
        [mesh],
        [material],
        [camera],
        [light],
        [parent, child],
        ambient=np.array([0.01, 0.02, 0.03], np.float32),
        name="sample",
        frames=10,
        segment_to=10,
    )


def test_scene_round_trip_behavioral_parity():
    original = sample_scene()
    restored = Scene.from_bytes(original.to_bytes())
    assert [mesh.name for mesh in restored.meshes] == ["quad"]
    np.testing.assert_array_equal(restored.meshes[0].vertices, original.meshes[0].vertices)
    np.testing.assert_array_equal(restored.meshes[0].faces, original.meshes[0].faces)
    np.testing.assert_array_equal(
        restored.meshes[0].smoothing, original.meshes[0].smoothing
    )
    np.testing.assert_array_equal(restored.meshes[0].texcoords, original.meshes[0].texcoords)
    assert restored.meshes[0].face_materials == ["paint", "paint"]
    assert restored.meshes[0].object_flags == 0x41
    assert restored.materials[0].name == "paint"
    np.testing.assert_allclose(restored.materials[0].ambient, original.materials[0].ambient, atol=1 / 255)
    np.testing.assert_allclose(
        restored.materials[0].diffuse, original.materials[0].diffuse, atol=1 / 255
    )
    np.testing.assert_allclose(restored.materials[0].specular, original.materials[0].specular, atol=1 / 255)
    assert restored.materials[0].shininess == pytest.approx(0.42, abs=0.005)
    assert restored.materials[0].transparency == pytest.approx(0.17, abs=0.005)
    assert restored.materials[0].two_sided
    assert restored.materials[0].texture1_map.name == "grid.png"
    np.testing.assert_array_equal(restored.materials[0].texture1_map.scale, [2, 3])
    np.testing.assert_array_equal(
        restored.materials[0].texture1_map.offset, [0.25, 0.5]
    )
    np.testing.assert_array_equal(restored.cameras[0].position, [4, 5, 6])
    np.testing.assert_array_equal(restored.cameras[0].target, [0, 0, 0])
    assert restored.cameras[0].roll == pytest.approx(0.2)
    assert restored.cameras[0].fov == pytest.approx(55, rel=1e-6)
    assert restored.cameras[0].see_cone
    assert restored.cameras[0].near_range == pytest.approx(0.1)
    assert restored.cameras[0].far_range == pytest.approx(100)
    np.testing.assert_array_equal(restored.lights[0].position, [3, 2, 8])
    np.testing.assert_allclose(restored.lights[0].color, [1, 0.8, 0.6])
    assert restored.lights[0].multiplier == pytest.approx(1.5)
    assert restored.lights[0].spot_light
    np.testing.assert_array_equal(restored.lights[0].spot, [0, 0, 0])
    assert restored.lights[0].hot_spot == pytest.approx(30)
    assert restored.lights[0].fall_off == pytest.approx(45)
    assert restored.lights[0].shadowed
    assert [node.parent_id for node in restored.nodes] == [65535, 10]


def test_writer_is_readable_by_assimp(tmp_path):
    path = tmp_path / "sample.3ds"
    subject = sample_scene()
    Scene(meshes=subject.meshes, materials=subject.materials).save(path)
    with assimp_load(str(path)) as reference:
        assert len(reference.meshes) == 1
        mesh = reference.meshes[0]
        assert len(mesh.faces) == 2
        assert len(mesh.vertices) >= 4
        expected = {
            tuple(vertex) for vertex in sample_scene().meshes[0].vertices[:4]
        }
        actual = {tuple(vertex) for vertex in np.asarray(mesh.vertices)}
        assert expected <= actual


def test_file_load_and_save(tmp_path):
    path = tmp_path / "roundtrip.3ds"
    sample_scene().save(path)
    restored = Scene.load(path)
    assert restored.name == "sample"
    assert restored.frames == 10


def test_node_hierarchy_evaluation():
    scene = sample_scene()
    matrices = scene.evaluate(5)
    np.testing.assert_allclose(matrices[0][3, :3], [2, 0, 0])
    np.testing.assert_allclose(matrices[1][3, :3], [2, 2, 0])


def test_vector_tcb_track_matches_upstream_coefficients():
    track = Track(
        [
            Key(0, np.array([0, 0, 0], np.float32)),
            Key(
                10,
                np.array([10, 0, 0], np.float32),
                tension=0.2,
                continuity=0.3,
                bias=-0.4,
            ),
            Key(20, np.array([0, 0, 0], np.float32)),
        ]
    )
    # Hermite at u=.5 with lib3ds first-key dd=10 and middle-key ds=-5.6.
    np.testing.assert_allclose(
        track.evaluate(5, np.zeros(3, np.float32)), [6.95, 0, 0], atol=2e-6
    )


def test_rotation_keys_round_trip_as_cumulative_quaternions():
    identity = np.array([0, 0, 0, 1], np.float32)
    quarter_turn = np.array(
        [0, 0, -np.sqrt(0.5), np.sqrt(0.5)], dtype=np.float32
    )
    node = Node(
        "rotating",
        node_id=2,
        rotation_track=Track([Key(0, identity), Key(10, quarter_turn)]),
        scale_track=Track([Key(0, np.ones(3, np.float32))]),
    )
    restored = Scene.from_bytes(Scene(nodes=[node]).to_bytes()).nodes[0]
    np.testing.assert_allclose(
        restored.rotation_track.keys[1].value, quarter_turn, atol=2e-6
    )


def test_node_input_order_is_not_required():
    scene = sample_scene()
    scene.nodes.reverse()
    matrices = scene.evaluate(0)
    np.testing.assert_allclose(matrices[0][3, :3], [1, 2, 0])


def test_cycle_is_rejected():
    scene = Scene(nodes=[Node("a", node_id=1, parent_id=2), Node("b", node_id=2, parent_id=1)])
    with pytest.raises(ValueError, match="cycle"):
        scene.evaluate(0)


@pytest.mark.parametrize(
    "payload",
    [
        b"",
        b"\x00" * 5,
        struct.pack("<HI", 0x1234, 6),
        struct.pack("<HI", 0x4D4D, 5),
        struct.pack("<HI", 0x4D4D, 12) + struct.pack("<HI", 1, 5),
        struct.pack("<HI", 0x4D4D, 20) + b"\0" * 6,
    ],
)
def test_malformed_files_are_rejected(payload):
    with pytest.raises(FormatError):
        Scene.from_bytes(payload)


def test_truncated_known_chunk_is_reported_as_format_error():
    child = struct.pack("<HI", 0x3D3E, 6)
    mdata = struct.pack("<HI", 0x3D3D, len(child) + 6) + child
    payload = struct.pack("<HI", 0x4D4D, len(mdata) + 6) + mdata
    with pytest.raises(FormatError):
        Scene.from_bytes(payload)


def test_empty_scene_and_empty_mesh_roundtrip():
    empty = Mesh(
        "empty",
        np.empty((0, 3), np.float32),
        np.empty((0, 3), np.int64),
    )
    restored = Scene.from_bytes(Scene(meshes=[empty]).to_bytes())
    assert restored.meshes[0].vertices.shape == (0, 3)
    assert restored.meshes[0].faces.shape == (0, 3)


def test_writer_rejects_mesh_beyond_3ds_count_limit():
    oversized = Mesh(
        "oversized",
        np.zeros((65536, 3), dtype=np.float32),
        np.empty((0, 3), dtype=np.int64),
    )
    with pytest.raises(ValueError, match="uint16"):
        Scene(meshes=[oversized]).to_bytes()
