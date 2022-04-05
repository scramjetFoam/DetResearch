import os
from unittest import TestCase, mock

import numpy as np
from uncertainties import unumpy as unp

from scripts.simulation_measurement_comparison.utilities.foil_image import (
    api,
    calculation_functions as cf,
    exceptions as ex,
)

THIS_MODULE = (
    "scripts.simulation_measurement_comparison.utilities.foil_image.api"
)


class TestCollectShotDeltas(TestCase):
    def setUp(self) -> None:
        self.shot_dir = "asdf"

    @mock.patch(f"{THIS_MODULE}.os.path.exists")
    def test_dir0_image_does_not_exist(
        self,
        path_exists_mock,
    ):
        # this should be the first call
        path_exists_mock.return_value = False

        with self.assertRaises(ex.ImageProcessingError) as err:
            bad_img_path = os.path.join(self.shot_dir, "dir0.png")
            good_err = f"{bad_img_path} does not exist"
            api.collect_shot_deltas(shot_dir=self.shot_dir)

            self.assertEqual(str(err), good_err)

    @mock.patch(f"{THIS_MODULE}.calculation_functions")
    @mock.patch(f"{THIS_MODULE}.os.path.exists")
    def test_dir1_image_does_not_exist(
        self,
        path_exists_mock,
        _,  # don't actually try to load an image
    ):
        # not the first call -- need side effect
        path_exists_mock.side_effect = [True, False]

        with self.assertRaises(ex.ImageProcessingError) as err:
            bad_img_path = os.path.join(self.shot_dir, "dir1.png")
            good_err = f"{bad_img_path} does not exist"
            api.collect_shot_deltas(shot_dir=self.shot_dir)

            self.assertEqual(str(err), good_err)

    @mock.patch(f"{THIS_MODULE}.calculation_functions")
    @mock.patch(f"{THIS_MODULE}.os.path.exists")
    def test_without_exclusion_images(
        self,
        path_exists_mock,
        cf_mock,
    ):
        dir0_img = np.array([
            [0, 1, 0, 1, 0],
            [1, 0, 0, 0, 1],
        ])
        dir1_img = np.array([
            [1, 1, 1, 1, 1],
            [1, 0, 1, 0, 1],
        ])
        good_0 = [2, 4]
        good_1 = [2, 2]
        good = np.array(good_0 + good_1)

        # directional images exist, exclusion images don't
        path_exists_mock.side_effect = [True, True, False, False]
        cf_mock.get_px_deltas_from_lines.side_effect = (
            cf.get_px_deltas_from_lines
        )
        cf_mock.load_image.side_effect = [dir0_img, dir1_img]

        test = api.collect_shot_deltas(shot_dir=self.shot_dir)

        np.testing.assert_array_equal(unp.nominal_values(test), good)

    @mock.patch(f"{THIS_MODULE}.calculation_functions")
    @mock.patch(f"{THIS_MODULE}.os.path.exists")
    def test_with_exclusion_images(
        self,
        path_exists_mock,
        cf_mock,
    ):
        dir0_img = np.array([
            [0, 1, 0, 1, 0],
            [1, 0, 0, 0, 1],
        ])
        exclusion_0 = np.array([
            [0, 0, 1, 0, 0],
            [0, 0, 0, 0, 0]
        ])
        dir1_img = np.array([
            [1, 1, 1, 1, 1],
            [1, 0, 1, 0, 1],
        ])
        exclusion_1 = np.array([
            [0, 0, 0, 0, 0],
            [1, 0, 0, 0, 0]
        ])
        good_0 = [4]
        good_1 = [2]
        good = np.array(good_0 + good_1)

        # all images exist
        path_exists_mock.side_effect = [True, True, True, True]
        cf_mock.get_px_deltas_from_lines.side_effect = (
            cf.get_px_deltas_from_lines
        )
        cf_mock.load_image.side_effect = [
            dir0_img,
            dir1_img,
            exclusion_0,
            exclusion_1,
        ]

        test = api.collect_shot_deltas(shot_dir=self.shot_dir)

        np.testing.assert_array_equal(unp.nominal_values(test), good)
