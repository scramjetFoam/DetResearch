"""
API functions for new soot foil image processing module

Anything important in here should be manually imported into __init__.py and
included in __all__ in order to keep the user interface clean.
"""

import os

import numpy as np

from . import calculation_functions, exceptions


class Shot:
    def __init__(
        self,
        date,
        shot_no,
    ):
        """
        A cleaner-than-a-tuple organizer for date/shot info

        Parameters
        ----------
        date: str
            Shot date
        shot_no: int
            Shot number
        """
        self.date = date
        self.shot_no = shot_no

    def __repr__(self):
        return f"{self.date} {self.shot_string}"

    @property
    def shot_string(self):
        return f"Shot {self.shot_no:02d}"

    @shot_string.setter
    def shot_string(self, _):
        raise AttributeError("shot_string cannot be set")

    @property
    def dir_name(self):
        return os.path.join(self.date, self.shot_string)

    @dir_name.setter
    def dir_name(self, _):
        raise AttributeError("dir_name cannot be set")

    def __len__(self):
        return len(str(self))


def collect_shot_deltas(
    shot_dir,
    use_exclusion_if_available=True,
):
    """
    Reads in images in a given directory and calculates all deltas, with
    exclusion zones accounted for (if the files exist).

    Parameters
    ----------
    shot_dir: str
        Directory where images for this shot are located.
    use_exclusion_if_available: bool
        Optionally ignore exclusion zones even if they _do_ exist.

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
        dir0_img = calculation_functions.load_image(dir0_path)
    else:
        raise exceptions.ImageProcessingError(f"{dir0_path} does not exist")
    if os.path.exists(dir1_path):
        dir1_img = calculation_functions.load_image(dir1_path)
    else:
        raise exceptions.ImageProcessingError(f"{dir1_path} does not exist")

    # exclusion zones are optional
    if os.path.exists(trash0_path) and use_exclusion_if_available:
        trash0_img = calculation_functions.load_image(trash0_path)
    else:
        trash0_img = np.zeros_like(dir0_img)
    if os.path.exists(trash1_path) and use_exclusion_if_available:
        trash1_img = calculation_functions.load_image(trash1_path)
    else:
        trash1_img = np.zeros_like(dir1_img)

    deltas0 = calculation_functions.get_px_deltas_from_lines(
        lines_img_in=dir0_img,
        exclusion_img_in=trash0_img,
    )
    deltas1 = calculation_functions.get_px_deltas_from_lines(
        lines_img_in=dir1_img,
        exclusion_img_in=trash1_img,
    )
    deltas = np.hstack((deltas0, deltas1))

    return deltas
