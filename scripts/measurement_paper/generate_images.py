#!/usr/bin/env python
# coding: utf-8

# # Strategize for rewrite
# This paper is being re-written. It was initially going to be based on the work that I had done with propane-air mixtures, however being as close as we were to the limits of detonation within our tube I had to make a last-minute pivot to methane-nitrous. During this pivot I also got a lot more in-depth with the schliren and soot foil analyses. In particular, the soot foil analysis has been changed to include spectral analysis methods, which will now be compared and contrasted with the original intended method.

# In[1]:


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
from funcs.post_processing.images.soot_foil.deltas import get_px_deltas_from_lines

d_drive = funcs.dir.d_drive


# plot formatting

# In[2]:


sns.set(style='whitegrid')
sns.set_color_codes("deep")
sns.set_context('paper')
sns.set_style({
    'font.family': 'serif', 
    'font.serif': 'Computer Modern',
})
# plt.rcParams["axes.titleweight"] = "bold"
plt.rcParams['figure.dpi'] = 200
cmap = "Greys_r"


# In[3]:


def sf_imread(img_path, plot=True):
    img_in = io.imread(img_path)
    if plot:
        img_in = transform.rotate(img_in, -90)  # show images going left-right
    return img_in


# In[4]:


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
        sf_delta_mm/sf_delta_px,
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


# ## Schlieren Measurements

# Read in data

# In[5]:


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


# In[6]:


[k for k in df_schlieren_frames if "u_" in k]


# ### Calculate cell size measurements

# In[7]:


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
    if(len(_df_this_shot)):
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
    
df_schlieren_tube = df_schlieren_tube[~pd.isna(df_schlieren_tube["cell_size"])]
df_schlieren_tube.head()


# ### Preprocessing

# Raw image

# In[8]:


schlieren_date = "2020-08-07"
schlieren_shot = 3
schlieren_frame = 0
schlieren_group = "fffff"
with pd.HDFStore(
    f"/d/Data/Processed/Data/data_{schlieren_group}.h5", 
    "r"
) as store:
    schlieren_key_date = schlieren_date.replace("-", "_")
    key = f"/schlieren/d{schlieren_key_date}/"           f"shot{schlieren_shot:02d}/"           f"frame_{schlieren_frame:02d}"
    schlieren_raw = np.fliplr(store[key])


# In[9]:


fig, ax = plt.subplots(figsize=(2, 3))
ax.imshow(schlieren_raw, cmap=cmap)
ax.axis("off")
ax.set_title("Raw")
ax.grid(False)


# Measurements

# In[10]:


fig, ax = plt.subplots(figsize=(2, 3))
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


# Spatial Calibration

# ### Delta Method

# Remove outliers

# In[11]:


schlieren_meas = df_schlieren_tube["cell_size"]
schlieren_meas = schlieren_meas[
    (schlieren_meas.mean() - 1.5*schlieren_meas.std() <= schlieren_meas) &
    (schlieren_meas <= schlieren_meas.mean() + 1.5*schlieren_meas.std())
]
n_schlieren_meas = len(schlieren_meas)
# cell_size_meas_foil + cell_size_uncert_foil
cell_size_meas_schlieren = schlieren_meas.mean()
cell_size_uncert_schlieren = (
    schlieren_meas.std() / 
    np.sqrt(n_schlieren_meas) * t.ppf(0.975, n_schlieren_meas-1)
)
print(f"{cell_size_meas_schlieren:.2f} +/- {cell_size_uncert_schlieren:.2f} mm")


# Plot

# In[51]:


fig, ax = plt.subplots(figsize=(6, 1.5))
sns.distplot(
    df_schlieren_tube["cell_size"],
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
    schlieren_meas.mean(),
    c="k",
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


# In[13]:


fig, ax = plt.subplots(figsize=(6, 1.5))
n_meas = np.arange(1, n_schlieren_meas+1)
running_mean = schlieren_meas.rolling(n_schlieren_meas, min_periods=0).mean()
running_std = schlieren_meas.rolling(n_schlieren_meas, min_periods=0).std()
running_sem = running_std / np.sqrt(n_meas)
plot_color = "C0"
plt.fill_between(
    n_meas, 
    running_mean + running_sem, running_mean - running_sem,
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
ax.set_ylabel("Mean Cell Size (mm)")
ax.set_title("Schlieren Cell Size Measurement")
ax.grid(False)
sns.despine()


# ## Soot Foil Measurements

# In[14]:


sf_date = "2020-12-27"
sf_shot = 3
sf_img_dir = os.path.join(
    d_drive,
    "Data",
    "Processed",
    "Soot Foil",
    "foil images",
    sf_date,
    f"Shot {sf_shot:02d}",
)
DF_SF_SPATIAL = pd.read_csv(
    os.path.join(
        d_drive,
        "Data",
        "Processed",
        "Soot Foil",
        "spatial_calibrations.csv"
    )
)
sf_spatial_shot_mask = (DF_SF_SPATIAL["date"] == sf_date) &     (DF_SF_SPATIAL["shot"] == sf_shot)
sf_delta_mm = DF_SF_SPATIAL[sf_spatial_shot_mask]["delta_mm"]
sf_delta_px = DF_SF_SPATIAL[sf_spatial_shot_mask]["delta_px"]
sf_scalebar = get_scale_bar(
    sf_delta_px,
    sf_delta_mm,
    cell_size=25.4,
)


# ### Preprocessing

# Raw soot foil images are pre-processed using RawTherapee and GIMP to straighten, de-skew, reduce noise, even out lighting, and emphasize cell boundaries. A spatial calibration is performed using a photographed ruler along with GIMP's measurement tool. Following calibration, a square portion of the image is selected for analysis.

# In[15]:


sf_img_raw = sf_imread(os.path.join(sf_img_dir, "square_raw.png"))
sf_img = sf_imread(os.path.join(sf_img_dir, "square.png"))
fig, ax = plt.subplots(1, 2, figsize=(6, 3))
ax[0].imshow(sf_img_raw, cmap=cmap)
ax[0].axis("off")
ax[0].set_title("Raw Image (Cropped)")
ax[1].imshow(sf_img, cmap=cmap)
ax[1].axis("off")
ax[1].set_title("Enhanced Image")
for a in ax:
    a.add_artist(copy(sf_scalebar),)
plt.tight_layout()
# Stoichiometric CH4-N2O with 37.2% molar N2 dilution
# left is raw, right is enhanced


# ### Delta Method

# #### Approach

# Manually outline cell boundaries

# In[16]:


sf_img_lines_thk = sf_imread(os.path.join(sf_img_dir, "lines_thk.png"))
fig, ax = plt.subplots(1, 2, figsize=(6, 3))
ax[0].imshow(sf_img, cmap=cmap)
ax[0].axis("off")
ax[0].set_title("Soot Foil")
ax[1].imshow(sf_img_lines_thk, cmap=cmap)
ax[1].axis("off")
ax[1].set_title("Traced Cells")
for a in ax:
    a.add_artist(copy(sf_scalebar),)
plt.tight_layout()


# Scan for triple point deltas

# In[17]:


sf_img_lines_z = sf_imread(os.path.join(sf_img_dir, "lines_zoomed.png"))
sf_img_lines_z = np.rot90(np.rot90(sf_img_lines_z)) # don't want to redo this
fig, ax = plt.subplots(figsize=(3, 3))
ax.imshow(sf_img_lines_z, cmap=cmap)
plt.axis("off")
plt.title("Soot Foil Measurement By Pixel Deltas")
arrow_x = 160
arrow_length = np.array([36, 32, 87, 52, 88, 35, 50])
arrow_y_top = np.array([-10, 20, 45, 126, 172, 254, 283])
n_arrows = len(arrow_length)
for i in range(n_arrows):
    if i == 0:
        arrowstyle = "-|>"
    elif i == n_arrows-1:
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


# #### Results

# ### Fix uncertainties

# Slightly modified edge pixel finding functions

# In[18]:


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


# Calculate uncertainties as needed and apply them to measurement dataframes

# In[19]:


df_sf_spatial = pd.read_csv("/d/Data/Processed/Soot Foil/spatial_calibrations.csv")
df_sf_spatial


# In[47]:


# add measurement pixel location precision uncertainty
# estimate using IMG_1983 (2020-12-27 Shot 03)
uncert_images_soot_foil = funcs.post_processing.images.schlieren.find_images_in_dir(
    "/d/Data/Processed/Soot Foil/foil images/2020-12-27/Shot 03/uncertainty/",
    ".png"
)
sf_repeatability_img_size = io.imread(uncert_images_soot_foil[0]).shape[0]  # get image size
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
_ = np.sqrt(np.sum(np.square(np.array([  # todo: apply this to actual measurements
    0.5,                # bias
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
u_px_cal_deltas = px_cal_deltas_soot_foil.std() /     np.sqrt(len(px_cal_deltas_soot_foil)) *     t.ppf(0.975, len(px_cal_deltas_soot_foil)-1)

# calculate and apply new calibration pixel uncertainty
# existing measurement accounts for sqrt2 from delta
# this applies directly without that because it is a direct delta measurement
df_sf_spatial["u_delta_px"] = np.sqrt(np.sum(np.square(np.array([
   df_sf_spatial["u_delta_px"],  # bias (preexisting)
    u_px_cal_deltas,             # precision (new)
]))))

# no need to do this for calibration mm uncertainty because it's a direct ruler
# reading, not a measurement of an existing quantity with a ruler
# (i.e. bias only)

plt.figure(figsize=(6, 1.5))
sns.distplot(px_cal_deltas_soot_foil, hist=False)
ax_ylim = plt.ylim()
plt.fill_between(
    [px_cal_deltas_soot_foil.mean() + u_px_cal_deltas,
     px_cal_deltas_soot_foil.mean() -u_px_cal_deltas],
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
plt.title(" Soot Foil Pixel Calibration Distance Repeatability Distribution")
plt.grid(False)
plt.xlabel("Ruler Distance (px)")
plt.ylabel("Probability\nDensity (1/px)")
sns.despine()


# In[21]:


from funcs.post_processing.images.soot_foil import deltas as pp_deltas
sf_lines_loc = os.path.join(sf_img_dir, "lines.png")
sf_img_lines = sf_imread(sf_lines_loc)
deltas = pp_deltas.get_px_deltas_from_lines(sf_lines_loc)
sf_cs_median = pp_deltas.get_cell_size_from_deltas(
    deltas,
    sf_delta_px,
    sf_delta_mm,
    np.median
).nominal_value
sf_cs_mean = pp_deltas.get_cell_size_from_deltas(
    deltas,
    sf_delta_px,
    sf_delta_mm,
    np.mean
).nominal_value
sf_cs_arr = pp_deltas.get_cell_size_from_deltas(
    deltas,
    sf_delta_px,
    sf_delta_mm,
    np.array
)
fig, ax = plt.subplots(figsize=(6, 1.5))
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
ax.set_title("Single Shot Measurement Distribution\nSoot Foil, Delta Method")
sns.despine()


# TODO:
#   * outline all repeatability foils
#   * collect all measurements
#   * do something about uncertainty!!!
#   * plot result for all repeatability foils

# Read in data

# In[22]:


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


# Crunch numbers

# In[38]:


measurements_foil = np.zeros(len(img_info)) * np.NaN

for idx, (date, shot) in enumerate(img_info):
    cal_mm, cal_px = DF_SF_SPATIAL[
        (DF_SF_SPATIAL["date"] == date) &
        (DF_SF_SPATIAL["shot"] == shot)
    ][["delta_mm", "delta_px"]].values[0]
    d_px = get_px_deltas_from_lines(
        f"/d/Data/Processed/Soot Foil/foil images/{date}/Shot {shot:02d}/composite.png",
        apply_uncertainty=False,
    )
    d_mm = d_px * cal_mm / cal_px
    measurements_foil[idx] = np.mean(d_mm)

# remove outliers
measurements_foil = measurements_foil[
    (measurements_foil <= measurements_foil.mean()+measurements_foil.std()*1.5) &
    (measurements_foil >= measurements_foil.mean()-measurements_foil.std()*1.5)
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
    np.sqrt(n_samples_foil) * t.ppf(0.975, n_samples_foil-1)
)

print(f"{cell_size_meas_foil:.2f} +/- {cell_size_uncert_foil:.2f} mm")


# In[46]:


fig, ax = plt.subplots(figsize=(6, 1.5))
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


# ## Comparison

# In[49]:


fig, ax = plt.subplots(figsize=(6, 1.5))
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


# Ratio check

# In[41]:


cell_size_meas_schlieren / cell_size_meas_foil, cell_size_meas_foil / cell_size_meas_schlieren

