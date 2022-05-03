import os
import sys
from copy import copy
from functools import wraps
from time import time

import skimage.filters

import funcs
import numpy as np
import pandas as pd
import seaborn as sns
import uncertainties as un
from funcs.post_processing.images.soot_foil import deltas as pp_deltas
from matplotlib import patches
from matplotlib import pyplot as plt
from matplotlib_scalebar.scalebar import ScaleBar
from scipy.stats import ks_2samp, t, ttest_ind, ttest_ind_from_stats
from skimage import io, transform
from uncertainties import unumpy as unp
import generate_images
from scipy.stats import linregress


def main():
    image_location = os.path.join(
        generate_images.SF_IMG_DIR,
        "composite.png",
    )
    composite_image = io.imread(image_location, as_gray=True)
    actual_image_rows = composite_image.shape[1]
    deltas = []
    distance = []
    for i in range(composite_image.shape[0]):
        row_deltas = pp_deltas._get_measurement_from_row(
            composite_image[i],
            255,
        )
        deltas.extend(row_deltas)
        distance.extend([i % actual_image_rows] * len(row_deltas))

    deltas = np.array(deltas)
    distance = np.array(distance)

    plt.figure()
    regression_all = linregress(distance, deltas)
    plt.plot(distance, deltas, label="deltas", marker=".", ls="", alpha=0.4)
    plt.plot(
        distance,
        regression_all.slope * distance + regression_all.intercept,
        label=f"regression R^2={regression_all.rvalue ** 2}",
    )
    plt.ylabel("triple point delta (px)")
    plt.xlabel("foil position (px)")
    plt.title("all measurements")
    plt.legend()

    plt.figure()
    delta_series = pd.Series(index=distance, data=deltas)
    delta_series = delta_series.groupby(delta_series.index).median()
    regression_median = linregress(delta_series.index, delta_series.values)
    plt.plot(
        delta_series.index,
        delta_series.values,
        label="deltas",
        marker=".",
        ls="",
        alpha=0.4,
    )
    plt.plot(
        delta_series.index,
        regression_median.slope * delta_series.index
        + regression_median.intercept,
        label=f"regression R^2={regression_median.rvalue ** 2}",
    )
    plt.ylabel("triple point delta (px)")
    plt.xlabel("foil position (px)")
    plt.title("median measurements")
    plt.legend()

    plt.show()


if __name__ == "__main__":
    main()
