"""
Modified tools from funcs

* exclusion zones are now included
* functions are a bit purer
* individual directional images are loaded rather than composite

If you're using this to calculate stuff, you probably want to just stick to
the api. Why are you in here?

todo: add tests
"""

import numpy as np
from skimage import io, color
from uncertainties import unumpy as unp
from funcs.uncertainty import add_uncertainty_terms, u_cell

from . import exceptions

u_cell = u_cell["soot_foil"]


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
    return binarize_array(color.rgb2gray(io.imread(img_path)))


def binarize_array(arr_in):
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
        raise exceptions.ImageProcessingError(
            "cannot process image with max <= 0"
        )

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
        raise exceptions.ImageProcessingError(
            f"image shape mismatch: "
            f"{exclusion_img_in.shape} vs. {lines_img.shape}"
        )
    else:
        exclusion_img = exclusion_img_in.copy()

    # NaN out exclusion zones
    lines_img = np.where(exclusion_img == 1, np.NaN, lines_img)

    deltas = []
    for i in range(lines_img.shape[0]):
        deltas.extend(get_diffs_from_row(lines_img[i]))

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


def get_diffs_from_sub_row(sub_row):
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

    return cell_boundary_index_diffs


def get_diffs_from_row(row):
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
            diffs.extend(get_diffs_from_sub_row(sub_row))

    return np.array(diffs)
