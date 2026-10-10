"""Direct unit tests for the sky-mesh and background-box helpers in background.py.

These helpers gate every sky mesh that is read from, or written to, a FITS
product. They were previously reached only through full background estimation,
so a change in their arithmetic or their validation order could pass the suite
while corrupting a mesh. Each test pins one documented behaviour at the
boundary.
"""

import unittest

import numpy as np

from weightmask.background import (
    _auto_box_size,
    _mesh_node_coords,
    _positive_int,
    _sky_mesh_shape,
    _validated_sky_mesh,
)


class TestPositiveInt(unittest.TestCase):
    def test_accepts_plain_and_numpy_integers(self):
        self.assertEqual(_positive_int(1, "box"), 1)
        self.assertEqual(_positive_int(np.int64(37), "box"), 37)
        self.assertIsInstance(_positive_int(np.int32(5), "box"), int)

    def test_rejects_booleans(self):
        for value in (True, False, np.bool_(True)):
            with self.subTest(value=value):
                with self.assertRaises(ValueError):
                    _positive_int(value, "box")

    def test_rejects_non_integers(self):
        for value in (1.5, 3.0, "4", None, [1]):
            with self.subTest(value=value):
                with self.assertRaises(ValueError):
                    _positive_int(value, "box")

    def test_rejects_zero_and_negative(self):
        for value in (0, -1, np.int64(-9)):
            with self.subTest(value=value):
                with self.assertRaises(ValueError):
                    _positive_int(value, "box")

    def test_message_names_the_field(self):
        with self.assertRaises(ValueError) as raised:
            _positive_int(0, "Sky mesh box size")
        self.assertIn("Sky mesh box size", str(raised.exception))


class TestSkyMeshShape(unittest.TestCase):
    def test_returns_image_shape_and_node_counts(self):
        (h, w), (ny, nx), box = _sky_mesh_shape((100, 100), 32)
        self.assertEqual((h, w), (100, 100))
        self.assertEqual((ny, nx), (4, 4))  # (99 // 32) + 1
        self.assertEqual(box, 32)

    def test_node_counts_are_unequal_for_a_non_square_image(self):
        (_h, _w), (ny, nx), _box = _sky_mesh_shape((65, 33), 32)
        self.assertEqual((ny, nx), (3, 2))

    def test_a_detector_narrower_than_the_box_keeps_one_node(self):
        (_h, _w), (ny, nx), _box = _sky_mesh_shape((30, 30), 32)
        self.assertEqual((ny, nx), (1, 1))

    def test_rejects_a_shape_that_is_not_two_dimensional(self):
        for shape in (5, (5,), (4, 4, 4)):
            with self.subTest(shape=shape):
                with self.assertRaises(ValueError):
                    _sky_mesh_shape(shape, 32)

    def test_rejects_zero_and_negative_dimensions(self):
        for shape in ((0, 10), (10, 0), (-4, 10)):
            with self.subTest(shape=shape):
                with self.assertRaises(ValueError):
                    _sky_mesh_shape(shape, 32)

    def test_rejects_an_invalid_box(self):
        for box in (0, -32, 16.5, "32", True):
            with self.subTest(box=box):
                with self.assertRaises(ValueError):
                    _sky_mesh_shape((100, 100), box)

    def test_accepts_numpy_integer_dimensions(self):
        shape, nodes, box = _sky_mesh_shape(np.array([64, 96]), 32)
        self.assertEqual(shape, (64, 96))
        self.assertEqual(nodes, (2, 3))
        self.assertEqual(box, 32)


class TestValidatedSkyMesh(unittest.TestCase):
    def test_accepts_a_finite_mesh_of_the_expected_shape(self):
        mesh = np.array([[5.0]], dtype=np.float32)
        data, shape, box = _validated_sky_mesh(mesh, (4, 4), 4)
        self.assertEqual(shape, (4, 4))
        self.assertEqual(box, 4)
        self.assertEqual(data.dtype, np.float64)
        self.assertTrue(data.flags["C_CONTIGUOUS"])

    def test_returns_input_unchanged_content(self):
        mesh = np.arange(6, dtype=np.float64).reshape(2, 3)
        # shape (8, 9) with box 4 -> nodes (2, 3)
        data, _shape, _box = _validated_sky_mesh(mesh, (8, 9), 4)
        np.testing.assert_allclose(data, mesh)

    def test_rejects_a_mesh_that_is_not_two_dimensional(self):
        for mesh in (np.zeros(4), np.zeros((1, 1, 1))):
            with self.subTest(rank=np.ndim(mesh)):
                with self.assertRaises(ValueError) as raised:
                    _validated_sky_mesh(mesh, (4, 4), 4)
                self.assertIn("two-dimensional", str(raised.exception))

    def test_rejects_non_numeric_mesh_data(self):
        with self.assertRaises(ValueError) as raised:
            _validated_sky_mesh(np.array([["x"]]), (4, 4), 4)
        self.assertIn("numeric", str(raised.exception))

    def test_rejects_non_finite_mesh_values(self):
        for bad in (np.nan, np.inf, -np.inf):
            with self.subTest(bad=bad):
                with self.assertRaises(ValueError) as raised:
                    _validated_sky_mesh(np.array([[bad]]), (4, 4), 4)
                self.assertIn("finite", str(raised.exception))

    def test_rejects_a_shape_mismatch(self):
        with self.assertRaises(ValueError) as raised:
            _validated_sky_mesh(np.ones((2, 2)), (4, 4), 4)
        self.assertIn("does not match expected shape", str(raised.exception))

    def test_validates_the_image_shape_before_the_mesh(self):
        # A bad image shape must fail even when the mesh itself is fine.
        with self.assertRaises(ValueError):
            _validated_sky_mesh(np.ones((1, 1)), (4,), 4)


class TestMeshNodeCoords(unittest.TestCase):
    def test_positions_follow_the_sep_node_phase(self):
        coords = _mesh_node_coords(4, 100, 32)
        # (k + 0.5) * 32 clipped to [0, 99]: the last node saturates at 99.
        np.testing.assert_allclose(coords, [16.0, 48.0, 80.0, 99.0])

    def test_a_single_node_sits_half_a_box_in(self):
        np.testing.assert_allclose(_mesh_node_coords(1, 100, 32), [16.0])

    def test_non_positive_node_count_yields_an_empty_array(self):
        for n in (0, -3):
            with self.subTest(n=n):
                coords = _mesh_node_coords(n, 100, 32)
                self.assertEqual(coords.shape, (0,))

    def test_coordinates_are_ordered_and_inside_the_image(self):
        for size, box in ((100, 32), (64, 16), (33, 32), (7, 4)):
            with self.subTest(size=size, box=box):
                n = (size - 1) // box + 1
                coords = _mesh_node_coords(n, size, box)
                self.assertEqual(len(coords), n)
                self.assertTrue(np.all(np.diff(coords) >= 0))
                self.assertGreaterEqual(coords.min(), 0.0)
                self.assertLessEqual(coords.max(), float(max(size - 1, 0)))

    def test_a_tiny_image_clips_every_node_to_the_same_pixel(self):
        np.testing.assert_allclose(_mesh_node_coords(3, 1, 32), [0.0, 0.0, 0.0])


class TestAutoBoxSize(unittest.TestCase):
    def test_scales_with_the_smaller_detector_dimension(self):
        # 2048/8 = 256 -> the auto ceiling; 64/8 = 8 -> the 32 px floor.
        self.assertEqual(_auto_box_size((2048, 4096), 0.0, {}), 256)
        self.assertEqual(_auto_box_size((64, 64), 0.0, {}), 32)
        self.assertEqual(_auto_box_size((512, 512), 0.0, {}), 64)

    def test_uses_the_smaller_dimension(self):
        self.assertEqual(_auto_box_size((4096, 256), 0.0, {}), _auto_box_size((256, 4096), 0.0, {}))

    def test_crowded_fields_double_the_box_up_to_a_ceiling(self):
        uncrowded = _auto_box_size((512, 512), 0.5, {})
        crowded = _auto_box_size((512, 512), 0.51, {})
        self.assertEqual(crowded, 2 * uncrowded)
        # A large detector doubles from 256 to the 512 ceiling, not to 512+.
        self.assertEqual(_auto_box_size((8192, 8192), 0.9, {}), 512)

    def test_config_box_is_a_ceiling_not_a_target(self):
        # Config cannot raise the box above what the image scale allows ...
        self.assertEqual(_auto_box_size((2048, 2048), 0.0, {"box_size": 4096}), 256)
        # ... but it can lower it.
        self.assertEqual(_auto_box_size((2048, 2048), 0.0, {"box_size": 50}), 50)

    def test_never_returns_below_sixteen(self):
        self.assertEqual(_auto_box_size((2048, 2048), 0.0, {"box_size": 4}), 16)
        self.assertEqual(_auto_box_size((8, 8), 0.0, {"box_size": 1}), 16)

    def test_unconvertible_config_falls_back_to_the_auto_box(self):
        for value in ("junk", None, object()):
            with self.subTest(value=value):
                self.assertEqual(_auto_box_size((512, 512), 0.0, {"box_size": value}), 64)

    def test_a_config_float_is_truncated_toward_zero(self):
        self.assertEqual(_auto_box_size((2048, 2048), 0.0, {"box_size": 50.9}), 50)


if __name__ == "__main__":
    unittest.main()
