import funcs
from copy import copy
import os
from skimage import io, transform, color
from matplotlib import pyplot as plt
import pandas as pd
import numpy as np
from matplotlib_scalebar.scalebar import ScaleBar
from matplotlib import patches
import uncertainties
from uncertainties import unumpy as unp
import seaborn as sns
from matplotlib.patches import Circle
from scipy.stats import t
from funcs.post_processing.images.soot_foil import deltas as pp_deltas

d_drive = funcs.dir.d_drive
DF_SF_SPATIAL = pd.read_csv(
    os.path.join(
        d_drive,
        "Data",
        "Processed",
        "Soot Foil",
        "spatial_calibrations.csv"
    )
)
SF_DATE = "2020-12-27"
SF_SHOT = 3
SF_IMG_DIR = os.path.join(
    d_drive,
    "Data",
    "Processed",
    "Soot Foil",
    "foil images",
    SF_DATE,
    f"Shot {SF_SHOT:02d}",
)
SF_SPATIAL_SHOT_MASK = (DF_SF_SPATIAL["date"] == SF_DATE) & \
                       (DF_SF_SPATIAL["shot"] == SF_SHOT)
SF_DELTA_MM = DF_SF_SPATIAL[SF_SPATIAL_SHOT_MASK]["delta_mm"]
SF_DELTA_PX = DF_SF_SPATIAL[SF_SPATIAL_SHOT_MASK]["delta_px"]


def set_plot_format():
    sns.set_color_codes("deep")
    sns.set_context('paper')
    sns.set_style({
        'font.family': 'serif',
        'font.serif': 'Computer Modern',
    })
    # plt.rcParams["axes.titleweight"] = "bold"
    plt.rcParams['figure.dpi'] = 200


def sf_imread(
        img_path,
        plot=True,
):
    img_in = io.imread(img_path)
    if plot:
        img_in = transform.rotate(img_in, -90)  # show images going left-right
    return img_in


def get_scale_bar(
    delta_px,
    delta_mm,
    cell_size,
    text_color="#000",
    box_color="#fff",
    box_alpha=0.75,
    rotation="vertical"
):
    return ScaleBar(
        delta_mm/delta_px,
        "mm",
        location=3,
        fixed_value=cell_size,
        scale_formatter=(lambda x, u: f"{x:.1f} {u}"),
        border_pad=0.2,
        color=text_color,
        box_color=box_color,
        box_alpha=box_alpha,
        rotation=rotation,
    )


def get_schlieren_data():
    # read in data
    df_schlieren_tube = pd.DataFrame()
    df_schlieren_frames = pd.DataFrame()
    for group in ("fffff", "hhhhh", "ggggg"):
        with pd.HDFStore(
                f"/d/Data/Processed/Data/data_{group}.h5",
                "r"
        ) as store:
            df_schlieren_tube = pd.concat((df_schlieren_tube, store.data))
        with pd.HDFStore(
                f"/d/Data/Processed/Data/schlieren_{group}.h5",
                "r"
        ) as store:
            df_schlieren_frames = pd.concat((df_schlieren_frames, store.data))

    # calculate cell size measurements
    df_schlieren_tube = df_schlieren_tube[
        np.isclose(df_schlieren_tube["phi_nom"], 1) &
        np.isclose(df_schlieren_tube["dil_mf_nom"], 0.2) &
        (df_schlieren_tube["fuel"] == "CH4") &
        (df_schlieren_tube["oxidizer"] == "N2O") &
        (df_schlieren_tube["diluent"] == "N2")
        ]
    df_schlieren_tube["cell_size"] = np.NaN
    df_schlieren_tube["u_cell_size"] = np.NaN

    for (date, shot), _ in df_schlieren_tube.groupby(["date", "shot"]):
        _df_this_shot = df_schlieren_frames[
            (df_schlieren_frames["date"] == date) &
            (df_schlieren_frames["shot"] == shot)
            ].dropna()
        if len(_df_this_shot):
            _deltas = unp.uarray(
                _df_this_shot["delta_px"],
                _df_this_shot["u_delta_px"]
            )
            _mm_per_px = unp.uarray(
                _df_this_shot["spatial_centerline"],
                _df_this_shot["u_spatial_centerline"]
            )
            _meas = np.mean(_deltas * _mm_per_px) * 2
            df_schlieren_tube.loc[
                (df_schlieren_tube["date"] == date) &
                (df_schlieren_tube["shot"] == shot),
                ["cell_size", "u_cell_size"]
            ] = _meas.nominal_value, _meas.std_dev

    df_schlieren_tube = df_schlieren_tube[
        ~pd.isna(df_schlieren_tube["cell_size"])]

    return df_schlieren_frames, df_schlieren_tube


def build_schlieren_images(
        cmap,
        df_schlieren_frames,
        image_width=None,
        image_height=None,
):
    aspect_ratio = 292/592  # w/h
    if image_width is None and image_height is None:
        raise ValueError("image_width or image_height must be given")

    if image_width is None:
        image_width = image_height * aspect_ratio
    elif image_height is None:
        image_height = image_width / aspect_ratio

    schlieren_date = "2020-08-07"
    schlieren_shot = 3
    schlieren_frame = 0
    schlieren_group = "fffff"
    with pd.HDFStore(
            f"/d/Data/Processed/Data/data_{schlieren_group}.h5",
            "r"
    ) as store:
        schlieren_key_date = schlieren_date.replace("-", "_")
        key = f"/schlieren/d{schlieren_key_date}/" \
              f"shot{schlieren_shot:02d}/" \
              f"frame_{schlieren_frame:02d}"
        schlieren_raw = np.fliplr(store[key])

    # raw frame
    fig, ax = plt.subplots(figsize=(image_width, image_height))
    fig.canvas.set_window_title("schlieren_frame_raw")
    ax.imshow(schlieren_raw, cmap=cmap)
    ax.axis("off")
    ax.set_title("Raw")
    ax.grid(False)
    plt.tight_layout()

    # measurements
    fig, ax = plt.subplots(figsize=(image_width, image_height))
    fig.canvas.set_window_title("schlieren_frame_measurements")
    ax.imshow(schlieren_raw, cmap=cmap)
    ax.axis("off")
    ax.set_title("Measurements")
    ax.grid(False)
    for loc_px in df_schlieren_frames[
        (df_schlieren_frames["date"] == schlieren_date) &
        (df_schlieren_frames["shot"] == schlieren_shot) &
        (df_schlieren_frames["frame"] == schlieren_frame)
    ]["loc_px"]:
        plt.axhline(
            loc_px,
            c="r",
            lw=0.5
        )
    plt.tight_layout()


def calculate_schlieren_cell_size(df_schlieren_tube):
    schlieren_meas = df_schlieren_tube["cell_size"]
    schlieren_meas = schlieren_meas[
        (schlieren_meas.mean() - 1.5 * schlieren_meas.std() <= schlieren_meas) &
        (schlieren_meas <= schlieren_meas.mean() + 1.5 * schlieren_meas.std())
        ]
    n_schlieren_meas = len(schlieren_meas)
    # cell_size_meas_foil + cell_size_uncert_foil
    cell_size_meas_schlieren = schlieren_meas.mean()
    cell_size_uncert_schlieren = (
            schlieren_meas.std() /
            np.sqrt(n_schlieren_meas) * t.ppf(0.975, n_schlieren_meas - 1)
    )

    return (
        cell_size_meas_schlieren,
        cell_size_uncert_schlieren,
        schlieren_meas,
        n_schlieren_meas,
    )


def plot_schlieren_measurement_distribution(
        schlieren_meas,
        cell_size_meas_schlieren,
        cell_size_uncert_schlieren,
        plot_width,
        plot_height,
):
    fig, ax = plt.subplots(figsize=(plot_width, plot_height))
    fig.canvas.set_window_title("schlieren_measurement_distribution")
    sns.distplot(
        schlieren_meas,
        hist=False,
        rug=True,
        ax=ax,
    )
    ax_ylim = ax.get_ylim()
    plt.fill_between(
        [cell_size_meas_schlieren + cell_size_uncert_schlieren,
         cell_size_meas_schlieren - cell_size_uncert_schlieren],
        ax_ylim[0],
        ax_ylim[1],
        alpha=0.25,
        color="k",
        ec=None,
        zorder=-1,
    )
    ax.axvline(
        cell_size_meas_schlieren,
        c="k",
        ls="--",
        alpha=0.7,
        zorder=-1,
    )
    ax.set_ylim(ax_ylim)
    ax.set_xlabel("Measured Cell Size (mm)")
    ax.set_ylabel("Probability Density\n(1/mm)")
    ax.set_title("Schlieren Cell Size Measurement Distribution")
    ax.grid(False)
    plt.tight_layout()
    sns.despine()


def plot_schlieren_measurement_convergence(
        schlieren_meas,
        n_schlieren_meas,
        plot_width,
        plot_height,
):
    fig, ax = plt.subplots(figsize=(plot_width, plot_height))
    fig.canvas.set_window_title("schlieren_measurement_convergence")
    n_meas = np.arange(1, n_schlieren_meas + 1)
    running_mean = schlieren_meas.rolling(
        n_schlieren_meas,
        min_periods=0,
    ).mean()
    running_std = schlieren_meas.rolling(
        n_schlieren_meas,
        min_periods=0,
    ).std()
    running_sem = running_std / np.sqrt(n_meas)
    plot_color = "C0"
    plt.fill_between(
        n_meas,
        running_mean + running_sem,
        running_mean - running_sem,
        alpha=0.25,
        color=plot_color,
        ec=None
    )
    plt.plot(
        n_meas,
        running_mean,
        "--",
        alpha=0.7,
        c=plot_color,
    )
    plt.scatter(
        n_meas,
        running_mean,
        c=plot_color,
        marker=".",
    )
    plt.xlim([2, len(running_mean)])
    ax.set_xlabel("Number of Frames Measured")
    ax.set_ylabel("Mean Cell Size\n(mm)")
    ax.set_title("Schlieren Cell Size Measurement")
    ax.grid(False)
    plt.tight_layout()
    sns.despine()


def build_soot_foil_images(
        cmap,
        image_height,
):
    # settings
    aspect_ratio = 2  # w/h
    image_width = aspect_ratio * image_height
    sf_scalebar = get_scale_bar(
        SF_DELTA_PX,
        SF_DELTA_MM,
        cell_size=25.4,
    )

    # read in foil images
    sf_img = sf_imread(os.path.join(SF_IMG_DIR, "square.png"))
    sf_img_lines_thk = sf_imread(os.path.join(SF_IMG_DIR, "lines_thk.png"))

    # display foil images
    fig, ax = plt.subplots(1, 2, figsize=(image_width, image_height))
    fig.canvas.set_window_title("soot_foil_images_main")
    ax[0].imshow(sf_img, cmap=cmap)
    ax[0].axis("off")
    ax[0].set_title("Soot Foil")
    ax[1].imshow(sf_img_lines_thk, cmap=cmap)
    ax[1].axis("off")
    ax[1].set_title("Traced Cells")
    for a in ax:
        a.add_artist(copy(sf_scalebar), )
    plt.tight_layout()

    # read in zoomed lines
    sf_img_lines_z = sf_imread(os.path.join(SF_IMG_DIR, "lines_zoomed.png"))
    sf_img_lines_z = np.rot90(
        np.rot90(sf_img_lines_z))  # don't want to redo this

    # plot zoomed lines
    fig, ax = plt.subplots(figsize=(image_height, image_height))
    fig.canvas.set_window_title("soot_foil_lines_zoomed")
    ax.imshow(sf_img_lines_z, cmap=cmap)
    plt.axis("off")
    plt.title("Soot Foil Measurement By Pixel Deltas")
    lines_scale = 900 / 330  # scaled up for quality
    arrow_x = 160 * lines_scale
    arrow_length = np.array([36, 32, 86, 52, 88, 35, 50]) * lines_scale
    arrow_y_top = np.array([-10, 20, 46, 126, 172, 254, 282]) * lines_scale
    n_arrows = len(arrow_length)
    for i in range(n_arrows):
        if i == 0:
            arrowstyle = "-|>"
        elif i == n_arrows - 1:
            arrowstyle = "<|-"
        else:
            arrowstyle = "<|-|>"
        arrow = patches.FancyArrowPatch(
            (arrow_x, arrow_y_top[i]),
            (arrow_x, arrow_y_top[i] + arrow_length[i]),
            arrowstyle=arrowstyle,
            mutation_scale=5,
            linewidth=0.75,
            color="r",
        )
        plt.gca().add_artist(arrow)
    plt.tight_layout()


def find_row_px_loc(row):
    row_locs = np.where(row == 255)[0]
    double_check = row_locs[
        np.abs(
            np.diff(
                [row_locs, np.roll(row_locs, -1)],
                axis=0
            )
        ).flatten() > 1
        ]
    if len(double_check):
        meas = double_check[0]
    else:
        meas = row_locs[0]
    return meas


def get_all_image_px_locs(img):
    return np.apply_along_axis(find_row_px_loc, 1, img)


def soot_foil_px_cal_uncertainty(
        plot_width,
        plot_height,
):
    # todo: figure out what's supposed to be going on here -- I think I have
    #  two different sets of uncertainty things happening and may only need the
    #  second
    # add measurement pixel location precision uncertainty
    # estimate using IMG_1983 (2020-12-27 Shot 03)
    img_dir = os.path.join(
        d_drive,
        "Data",
        "Processed",
        "Soot Foil",
        "foil images",
        "2020-12-27",
        "Shot 03",
        "uncertainty",
    )
    uncert_images_soot_foil = funcs.post_processing.images.schlieren.\
        find_images_in_dir(
            img_dir,
            ".png"
        )
    # noinspection PyUnresolvedReferences
    sf_repeatability_img_size = io.imread(
        uncert_images_soot_foil[0]
    ).shape[0]  # get image size
    # noinspection PyTypeChecker
    n_sf_repeatability_images = len(uncert_images_soot_foil)
    sf_repeatability_px_locs = np.ones((
        sf_repeatability_img_size,
        n_sf_repeatability_images,
    )) * np.NaN
    for i, img_loc in enumerate(uncert_images_soot_foil):
        img = io.imread(img_loc)
        sf_repeatability_px_locs[:, i] = get_all_image_px_locs(img)

    # use max std of all rows as uncertainty estimate
    u_px_loc_precision = np.std(
        sf_repeatability_px_locs,
        axis=1,
    ).max() / np.sqrt(n_sf_repeatability_images) * t.ppf(
        0.975,
        n_sf_repeatability_images - 1,
    )

    # calculate and apply new measurement pixel location precision uncertainty
    _ = np.sqrt(
        np.sum(np.square(np.array([  # todo: apply this to actual measurements
            0.5,  # bias
            u_px_loc_precision  # precision
        ])))) * np.sqrt(2)  # sqrt 2 to account for propagation in delta

    # add pixel delta calibration precision uncertainty
    # estimate using IMG_1983 (2020-12-27 Shot 03)
    px_cal_deltas_soot_foil = np.array([
        2344,  # this is what is saved in the .xcf
        2347,
        2345,
        2345,
        2345,
        2344,
        2344,
        2345,
        2344,
        2345,
    ])
    u_px_cal_deltas = px_cal_deltas_soot_foil.std() / \
        np.sqrt(len(px_cal_deltas_soot_foil)) * \
        t.ppf(0.975, len(px_cal_deltas_soot_foil) - 1)

    # calculate and apply new calibration pixel uncertainty
    # existing measurement accounts for sqrt2 from delta
    # this applies directly without that because it is a direct delta
    # measurement
    DF_SF_SPATIAL["u_delta_px"] = np.sqrt(np.sum(np.square(np.array([
        DF_SF_SPATIAL["u_delta_px"],  # bias (preexisting)
        u_px_cal_deltas,  # precision (new)
    ]))))

    # no need to do this for calibration mm uncertainty because it's a direct
    # ruler
    # reading, not a measurement of an existing quantity with a ruler
    # (i.e. bias only)

    fig = plt.figure(figsize=(plot_width, plot_height))
    fig.canvas.set_window_title("soot_foil_px_cal_uncertainty_distribution")
    sns.distplot(px_cal_deltas_soot_foil, hist=False)
    ax_ylim = plt.ylim()
    plt.fill_between(
        [px_cal_deltas_soot_foil.mean() + u_px_cal_deltas,
         px_cal_deltas_soot_foil.mean() - u_px_cal_deltas],
        ax_ylim[0],
        ax_ylim[1],
        alpha=0.25,
        color="k",
        ec=None,
        zorder=-1,
    )
    plt.axvline(
        px_cal_deltas_soot_foil.mean(),
        c="k",
        ls="--",
        alpha=0.7,
        zorder=-1,
    )
    plt.ylim(ax_ylim)
    plt.title(
        " Soot Foil Pixel Calibration Distance Repeatability Distribution")
    plt.grid(False)
    plt.xlabel("Ruler Distance (px)")
    plt.ylabel("Probability\nDensity (1/px)")
    sns.despine()
    plt.tight_layout()


def get_single_foil_delta_distribution(
        plot_width,
        plot_height,
):
    sf_lines_loc = os.path.join(SF_IMG_DIR, "lines.png")
    deltas = pp_deltas.get_px_deltas_from_lines(sf_lines_loc)
    sf_cs_mean = pp_deltas.get_cell_size_from_deltas(
        deltas,
        SF_DELTA_PX,
        SF_DELTA_MM,
        np.mean
    ).nominal_value
    sf_cs_arr = pp_deltas.get_cell_size_from_deltas(
        deltas,
        SF_DELTA_PX,
        SF_DELTA_MM,
        np.array
    )
    fig, ax = plt.subplots(figsize=(plot_width, plot_height))
    fig.canvas.set_window_title("single_foil_delta_distribution")
    sns.distplot(
        unp.nominal_values(sf_cs_arr),
        #     kde=False,
        hist=False,
        ax=ax,
    )
    ax.axvline(
        sf_cs_mean,
        color="k",
        ls="-",
        label=f"mean: {sf_cs_mean:8.1f} mm",
        alpha=0.75
    )
    ax.legend()
    ax.set_xlim([0, plt.xlim()[1]])
    ax.grid(False)
    ax.set_xlabel("Measurement (mm)")
    ax.set_ylabel("Count")
    ax.set_title(
        "Single Shot Measurement Distribution\nSoot Foil, Delta Method")
    sns.despine()
    plt.tight_layout()


def calculate_soot_foil_cell_size():
    img_info = (
        # date, shot
        ("2020-11-12", 0),
        ("2020-11-13", 8),
        ("2020-11-23", 3),
        ("2020-11-23", 4),
        ("2020-11-23", 6),
        ("2020-11-23", 7),
        ("2020-11-24", 0),
        ("2020-11-24", 3),
        ("2020-11-24", 7),
        ("2020-11-25", 0),
        ("2020-12-20", 8),
        ("2020-12-21", 9),
        ("2020-12-27", 0),
        ("2020-12-27", 1),
        ("2020-12-27", 2),
        ("2020-12-27", 3),
        ("2020-12-27", 6),
        ("2020-12-27", 7),
        ("2020-12-27", 8),
    )
    measurements_foil = np.zeros(len(img_info)) * np.NaN

    for idx, (date, shot) in enumerate(img_info):
        cal_mm, cal_px = DF_SF_SPATIAL[
            (DF_SF_SPATIAL["date"] == date) &
            (DF_SF_SPATIAL["shot"] == shot)
            ][["delta_mm", "delta_px"]].values[0]
        d_px = pp_deltas.get_px_deltas_from_lines(
            os.path.join(
                d_drive,
                "Data",
                "Processed",
                "Soot Foil",
                "foil images",
                f"{date}",
                f"Shot {shot:02d}",
                "composite.png",
            ),
            apply_uncertainty=False,
        )
        d_mm = d_px * cal_mm / cal_px
        measurements_foil[idx] = np.mean(d_mm)

    # remove outliers
    mean = measurements_foil.mean()
    std = measurements_foil.std()
    measurements_foil = measurements_foil[
        (measurements_foil <= mean + std * 1.5) &
        (measurements_foil >= mean - std * 1.5)
        ]

    # scale to match number of samples with schlieren
    # reduced_indices_foil = np.random.choice(
    #     np.arange(len(measurements_foil)),
    #     n_schlieren_meas,
    #     replace=False,
    # )
    # copy/paste result for consistency between runs of the notebook
    reduced_indices_foil = [0, 1, 2, 6, 7, 9, 10, 11, 12, 13, 14, 15, 16, 17]
    measurements_foil = measurements_foil[reduced_indices_foil]
    n_samples_foil = len(measurements_foil)

    cell_size_meas_foil = measurements_foil.mean()
    cell_size_uncert_foil = (
            measurements_foil.std() /
            np.sqrt(n_samples_foil) * t.ppf(0.975, n_samples_foil - 1)
    )

    return measurements_foil, cell_size_meas_foil, cell_size_uncert_foil


def plot_soot_foil_measurement_distribution(
        measurements_foil,
        cell_size_meas_foil,
        cell_size_uncert_foil,
        plot_width,
        plot_height,
):
    fig, ax = plt.subplots(figsize=(plot_width, plot_height))
    fig.canvas.set_window_title("soot_foil_measurement_distribution")
    sns.distplot(measurements_foil, hist=False, rug=True, ax=ax)
    ax.axvline(
        cell_size_meas_foil,
        color="k",
        ls="--",
        alpha=0.7,
        zorder=-1,
    )
    ax_ylim = ax.get_ylim()
    ax.fill_between(
        [cell_size_meas_foil + cell_size_uncert_foil,
         cell_size_meas_foil - cell_size_uncert_foil],
        ax_ylim[0],
        ax_ylim[1],
        alpha=0.25,
        color="k",
        ec=None,
        zorder=-1,
    )
    ax.set_ylim(ax_ylim)
    ax.set_xlabel("Cell Size (mm)")
    ax.set_ylabel("Probability Density\n(1/mm)")
    ax.set_title("Soot Foil Measurement Distribution")
    ax.grid(False)
    sns.despine()
    plt.tight_layout()


def plot_cell_size_comparison(
        schlieren_meas,
        cell_size_meas_schlieren,
        cell_size_uncert_schlieren,
        measurements_foil,
        cell_size_meas_foil,
        cell_size_uncert_foil,
        plot_width,
        plot_height,
):
    fig, ax = plt.subplots(figsize=(plot_width, plot_height))
    fig.canvas.set_window_title("cell_size_comparison")
    sns.distplot(  # schlieren
        schlieren_meas,
        hist=False,
        ax=ax,
        label="Schlieren",
    )
    sns.distplot(  # soot foil
        measurements_foil,
        hist=False,
        ax=ax,
        label="Soot Foil",
    )

    plt.legend(frameon=False)

    ax_ylim = ax.get_ylim()

    plt.fill_between(  # schlieren
        [cell_size_meas_schlieren + cell_size_uncert_schlieren,
         cell_size_meas_schlieren - cell_size_uncert_schlieren],
        ax_ylim[0],
        ax_ylim[1],
        alpha=0.25,
        color="C0",
        ec=None,
        zorder=-1,
    )
    ax.fill_between(  # soot foil
        [cell_size_meas_foil + cell_size_uncert_foil,
         cell_size_meas_foil - cell_size_uncert_foil],
        ax_ylim[0],
        ax_ylim[1],
        alpha=0.25,
        color="C1",
        ec=None,
        zorder=-1,
    )

    ax.axvline(  # schlieren
        cell_size_meas_schlieren,
        c="C0",
        ls="--",
        alpha=0.7,
        zorder=-1,
    )
    ax.axvline(  # soot foil
        cell_size_meas_foil,
        color="C1",
        ls="--",
        alpha=0.7,
        zorder=-1,
    )

    ax.set_ylim(ax_ylim)
    # ax.axvline(df_schlieren_tube["cell_size"].median())
    ax.set_xlabel("Measured Cell Size (mm)")
    ax.set_ylabel("Probability Density\n(1/mm)")
    ax.set_title("Schlieren Cell Size Measurement Distribution")
    ax.grid(False)
    sns.despine()
    plt.tight_layout()


def main():
    cmap = "Greys_r"
    plot_width = 6
    plot_height = 2
    image_height = 3
    set_plot_format()

    # do schlieren stuff
    df_schlieren_frames, df_schlieren_tube = get_schlieren_data()
    build_schlieren_images(
        cmap,
        df_schlieren_frames,
        image_height=image_height,
    )
    (cell_size_meas_schlieren,
     cell_size_uncert_schlieren,
     schlieren_meas,
     n_schlieren_meas) = calculate_schlieren_cell_size(df_schlieren_tube)
    plot_schlieren_measurement_distribution(
        schlieren_meas,
        cell_size_meas_schlieren,
        cell_size_uncert_schlieren,
        plot_width,
        plot_height,
    )
    plot_schlieren_measurement_convergence(
        schlieren_meas,
        n_schlieren_meas,
        plot_width,
        plot_height,
    )
    print(f"schlieren: "
          f"{cell_size_meas_schlieren:.2f}+/-"
          f"{cell_size_uncert_schlieren:.2f} mm")

    # do soot foil stuff
    build_soot_foil_images(
        cmap,
        image_height,
    )
    soot_foil_px_cal_uncertainty(  # THIS UPDATES DF_SF_SPATIAL
        plot_width,
        plot_height,
    )
    (measurements_foil,
     cell_size_meas_foil,
     cell_size_uncert_foil) = calculate_soot_foil_cell_size()
    plot_soot_foil_measurement_distribution(
        measurements_foil,
        cell_size_meas_foil,
        cell_size_uncert_foil,
        plot_width,
        plot_height,
    )
    print(f"soot foil: "
          f"{cell_size_meas_foil:.2f}+/-"
          f"{cell_size_uncert_foil:.2f} mm")

    # comparison
    plot_cell_size_comparison(
        schlieren_meas,
        cell_size_meas_schlieren,
        cell_size_uncert_schlieren,
        measurements_foil,
        cell_size_meas_foil,
        cell_size_uncert_foil,
        plot_width,
        plot_height,
    )
    foil_to_schlieren = uncertainties.ufloat(
        cell_size_meas_foil,
        cell_size_uncert_foil,
    ) / uncertainties.ufloat(
        cell_size_meas_schlieren,
        cell_size_uncert_schlieren,
    )
    print(f"soot foil/schlieren ratio: {foil_to_schlieren:.2f}")


if __name__ == "__main__":
    main()
    plt.show()
