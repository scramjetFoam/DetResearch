from unittest import TestCase

import numpy as np

from scripts.simulation_measurement_comparison.utilities.foil_image import (
    calculation_functions as cf,
    exceptions as ex,
)


class TestBinarizeArray(TestCase):
    def test_bad_value(self):
        """
        binarize_array raises an error on bad values
        """
        with self.subTest("zero"):
            with self.assertRaises(ex.ImageProcessingError):
                cf.binarize_array(np.array([0]))

        with self.subTest("negative"):
            with self.assertRaises(ex.ImageProcessingError):
                cf.binarize_array(np.array([0]))

        with self.subTest("NaN"):
            with self.assertRaises(ex.ImageProcessingError):
                with self.assertWarns(RuntimeWarning):
                    # numpy throws a runtime warning for all-NaN slice
                    cf.binarize_array(np.array([np.NaN]))

    def test_good_value(self):
        """
        binarize_array returns an array with only 0, 1
        """
        arr_in = np.array([0.5, 0.5, 0.0, 0.1])
        good = np.array([1, 1, 0, 0], dtype="int")

        test = cf.binarize_array(arr_in)

        with self.subTest("values"):
            np.testing.assert_allclose(test, good)

        with self.subTest("dtypes"):
            self.assertEqual(test.dtype, good.dtype)


class TestGetDiffsFromSubRow(TestCase):
    def test_good_value(self):
        """
        get_diffs_from_sub_row works properly
        """
        sub_row = np.array([np.NaN, 1, 0, 0, 1, 0, 1, 1, 1, 0, 1])
        good = np.array([3, 2, 2])

        test = cf.get_diffs_from_sub_row(sub_row)

        np.testing.assert_array_equal(test, good)


class TestGetDiffsFromRow(TestCase):
    def test_all_nan(self):
        """
        get_diffs_from_row handles all-NaN input
        """
        test = cf.get_diffs_from_row(np.array([np.NaN]))

        np.testing.assert_array_equal(test, np.array([]))

    def test_all_zero(self):
        """
        get_diffs_from_row handles all-zero input
        """
        test = cf.get_diffs_from_row(np.array([0]))

        np.testing.assert_array_equal(test, np.array([]))

    def test_good_value(self):
        """
        get_diffs_from_row works properly
        """
        row = np.array([1, 0, 0, 1, np.NaN, 1, 0, 1, np.NaN, np.NaN, 1, np.NaN])
        good = np.array([3, 2])

        test = cf.get_diffs_from_row(row)

        np.testing.assert_array_equal(test, good)


class TestGetPxDeltasFromLines(TestCase):
    def test_shape_mismatch(self):
        """
        get_px_deltas_from_lines raises an error when image shapes don't match
        """
        lines_img_in = np.array([1, 0])
        exclusion_img_in = np.array([1, 0, 1])

        with self.assertRaises(ex.ImageProcessingError):
            cf.get_px_deltas_from_lines(
                lines_img_in=lines_img_in,
                exclusion_img_in=exclusion_img_in,
            )

    def test_good_value_masked(self):
        """
        get_px_deltas_from_lines works properly
        """
        lines_img_in = np.array([
            [1, 0, 1, 0, 0, 1, 0, 1],
            [1, 0, 1, 0, 0, 1, 0, 1],
            [1, 0, 1, 0, 0, 1, 0, 1],
        ])
        exclusion_img_in = np.array([
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0, 0, 1],  # NaNs out both 0 and 1
            [0, 1, 1, 1, 1, 0, 0, 0],  # NaNs out both 0 and 1
        ])
        good = np.array([2, 3, 2, 3, 2])  # second row only has the 3 px gap

        test = cf.get_px_deltas_from_lines(
            lines_img_in=lines_img_in,
            exclusion_img_in=exclusion_img_in,
            apply_uncertainty=False,
        )

        np.testing.assert_array_equal(test, good)

    def test_good_value_unmasked(self):
        """
        get_px_deltas_from_lines works properly
        """
        lines_img_in = np.array([
            [1, 0, 1, 0, 0, 1, 0, 1],
            [1, 0, 1, 0, 0, 1, 0, 1],
        ])
        exclusion_img_in = np.array([
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
        ])
        good = np.array([2, 3, 2, 2, 3, 2])

        test = cf.get_px_deltas_from_lines(
            lines_img_in=lines_img_in,
            exclusion_img_in=exclusion_img_in,
            apply_uncertainty=False,
        )

        np.testing.assert_array_equal(test, good)
