import numpy as np
import uncertainties as un
from skimage import io, color
from uncertainties import unumpy as unp

from ....uncertainty import add_uncertainty_terms, u_cell
from ._rust import fast_get_px_deltas_from_lines as _fast_get_deltas

u_cell = u_cell["soot_foil"]


def get_px_deltas_from_lines(
        img_path,
        use_fast=True,
        mask_path=None,
        apply_uncertainty=True
):
    """
    Returns an array of per-row triple point deltas (in pixels) from a given
    image. The spatial calibration factor, mm_per_px, is required in order to
    remove deltas larger than the tube diameter, which are unphysical and a
    result of the limitations of soot foil measurement.

    Parameters
    ----------
    img_path : str
        path of the image containing traced triple point lines
    use_fast : bool
        If true, the faster compiled rust version of the processing script will be run
    mask_path : Optional[str]
        Path to the damage mask image, which must have the same dimensions as the main image. Only works if
        `use_fast=True`
    apply_uncertainty : bool
        True returns array of nominal values with uncertainties; False returns
        only nominal values

    Returns
    -------
    deltas : np.array or unp.uarray
        triple point distances, in pixels.
    """
    if use_fast:
        deltas = _fast_get_deltas(img_path, mask_path)
    else:
        img = color.rgb2gray(io.imread(img_path))
        img_max = img.max()
        deltas = []
        for i in range(img.shape[0]):
            deltas.extend(_get_measurement_from_row(img[i], img_max))

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


def _get_measurement_from_row(row, img_max):
    row_locs = np.where(row == img_max)[0]
    # get rid of adjacent pixels
    # if two measurements are in adjacent pixels, this method selects the
    # rightmost adjacent location and throws out the others (i.e. it only
    # accepts measurements where the location is >1 away from the previous)
    row_locs = row_locs[
        np.abs(
            np.diff(
                [row_locs, np.roll(row_locs, -1)],
                axis=0
            )
        ).flatten() > 1
    ]
    return np.diff(row_locs)


def get_cell_size_from_deltas(
        deltas,
        l_px_i,
        l_mm_i,
        estimator=np.median
):
    """
    Converts pixel triple point deltas to cell size

    Parameters
    ----------
    deltas : np.array or pandas.Series
    l_px_i : float
        nominal value of spatial calibration factor (px)
    l_mm_i : float
        nominal value of spatial calibration factor (mm)
    estimator : function
        function used to estimate cell size from triple point measurements

    Returns
    -------
    un.ufloat
        estimated cell size
    """
    l_px_i = un.ufloat(
        l_px_i,
        add_uncertainty_terms([
            u_cell["l_px"]["b"],
            u_cell["l_px"]["p"]
        ])
    )
    l_mm_i = un.ufloat(
        l_mm_i,
        add_uncertainty_terms([
            u_cell["l_mm"]["b"],
            u_cell["l_mm"]["p"]
        ])
    )
    return 2 * estimator(deltas) * l_mm_i / l_px_i
