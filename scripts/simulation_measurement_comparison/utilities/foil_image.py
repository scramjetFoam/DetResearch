"""
Modified tools from funcs:
* exclusion zones are now included
* functions are a bit purer
* individual directional images are loaded rather than composite

todo: add tests
todo: compare exclusion/non-exclusion measurements
"""

import os

import numpy as np
from skimage import io, color
from uncertainties import unumpy as unp

from funcs.uncertainty import add_uncertainty_terms, u_cell

u_cell = u_cell["soot_foil"]


class ImageProcessingError(Exception):
    pass


def collect_shot_deltas(shot_dir):
    """
    Reads in images in a given directory and calculates all deltas, with
    exclusion zones accounted for (if the files exist).

    Parameters
    ----------
    shot_dir: str
        Directory where images for this shot are located.

    Returns
    -------
    np.ndarray
        Array of measured deltas.
    """
    dir0_path = os.path.join(shot_dir, "dir0.png")
    dir1_path = os.path.join(shot_dir, "dir1.png")
    trash0_path = os.path.join(shot_dir, "trash0.png")
    trash1_path = os.path.join(shot_dir, "trash1.png")

    # lines are mandatory
    if os.path.exists(dir0_path):
        dir0_img = load_image(dir0_path)
    else:
        raise ImageProcessingError(f"{dir0_path} does not exist")
    if os.path.exists(dir1_path):
        dir1_img = load_image(dir1_path)
    else:
        raise ImageProcessingError(f"{dir1_path} does not exist")

    # exclusion zones are optional
    if os.path.exists(trash0_path):
        trash0_img = load_image(trash0_path)
    else:
        trash0_img = np.zeros_like(dir0_img)
    if os.path.exists(trash1_path):
        trash1_img = load_image(trash1_path)
    else:
        trash1_img = np.zeros_like(dir1_img)

    deltas0 = get_px_deltas_from_lines(
        lines_img_in=dir0_img,
        exclusion_img_in=trash0_img,
    )
    deltas1 = get_px_deltas_from_lines(
        lines_img_in=dir1_img,
        exclusion_img_in=trash1_img,
    )
    deltas = np.hstack((deltas0, deltas1))

    return deltas


def load_image(img_path):
    """
    Loads in an image as a black and white array

    Parameters
    ----------
    img_path: str
        Path of image to be loaded in.

    Returns
    -------

    """
    return _binarize_array(color.rgb2gray(io.imread(img_path)))


def _binarize_array(arr_in):
    """
    Converts an input array to have integer values of 1 at max, and 0
    everywhere else.

    Parameters
    ----------
    arr_in: np.ndarray
        Input array

    Returns
    -------
    np.ndarray
        Converted array (copy)
    """
    arr_max = arr_in.max()
    if arr_max <= 0:
        raise ImageProcessingError("cannot process image with max <= 0")

    arr_out = (arr_in.copy() / arr_in.max()).astype(int)
    arr_out[arr_out < 1] = 0

    return arr_out


def get_px_deltas_from_lines(
    lines_img_in,
    exclusion_img_in,
    apply_uncertainty=True,
):
    """
    Returns an array of per-row triple point deltas (in pixels) from a given
    image. The spatial calibration factor, mm_per_px, is required in order to
    remove deltas larger than the tube diameter, which are unphysical and a
    result of the limitations of soot foil measurement.

    Parameters
    ----------
    lines_img_in : np.ndarray
        Array containing traced triple point lines for a single direction.
    exclusion_img_in : np.ndarray
        Array containing traced triple point lines for a single direction.
    apply_uncertainty : bool
        True returns array of nominal values with uncertainties; False returns
        only nominal values.

    Returns
    -------
    deltas : np.ndarray
        Triple point distances, in pixels.
    """
    lines_img = lines_img_in.copy()

    if exclusion_img_in.shape != lines_img.shape:
        raise ImageProcessingError(
            f"image shape mismatch: "
            f"{exclusion_img_in.shape} vs. {lines_img.shape}"
        )
    else:
        exclusion_img = exclusion_img_in.copy()

    # NaN out exclusion zones
    lines_img = np.where(exclusion_img == 1, np.NaN, lines_img)

    deltas = []
    for i in range(lines_img.shape[0]):
        deltas.extend(_get_diffs_from_row(lines_img[i]))

    deltas = np.array(deltas)

    if apply_uncertainty:
        uncert = add_uncertainty_terms([
            u_cell["delta_px"]["b"],
            u_cell["delta_px"]["p"]
        ])
        deltas = unp.uarray(
            deltas,
            uncert
        )

    return deltas


def _get_diffs_from_sub_row(sub_row):
    """
    The actual diff-getter. If two measurements are in adjacent pixels, this
    method selects the rightmost adjacent location and throws out the others
    (i.e. it only accepts measurements where the boundary location is >1 px away
    from the previous boundary location).

    Parameters
    ----------
    sub_row: np.ndarray
        Portion of a single row of image pixels within a given exclusion zone.

    Returns
    -------
    np.ndarray
        Distances between cell boundary locations.
    """
    # locate cell boundaries
    cell_boundary_indices = np.where(sub_row == 1)[0]

    # find how far apart adjacent boundaries are
    cell_boundary_index_diffs = np.abs(
        cell_boundary_indices - np.roll(cell_boundary_indices, -1)
    )

    # throw out adjacent boundaries
    cell_boundary_index_diffs = (
        cell_boundary_index_diffs[cell_boundary_index_diffs > 1]
    )

    return np.diff(cell_boundary_index_diffs)


def _get_diffs_from_row(row):
    """
    Get all pixel distances between cell boundaries for a single row in an image

    Parameters
    ----------
    row: np.ndarray
        Row containing 1 where cell boundaries exist, NaN where exclusion zones
        exist, and 0 otherwise.

    Returns
    -------
    np.ndarray
        Concatenated array of cell boundary distances within exclusion zones.
    """
    diffs = []
    # skip rows without useful data
    if not (np.all(np.isnan(row)) or np.allclose(row, 0)):
        # split into sub-arrays on NaN to enforce exclusion zones
        split_row = np.split(row, np.where(np.isnan(row))[0])
        for sub_row in split_row:
            diffs.extend(_get_diffs_from_sub_row(sub_row))

    return np.array(diffs)
