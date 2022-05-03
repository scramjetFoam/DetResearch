import os

import generate_images
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt

import funcs

d_drive = funcs.dir.d_drive


def main():
    generate_images.set_plot_format()
    palette = [generate_images.COLOR_SF, generate_images.COLOR_SC, "k"]
    df_comparison = pd.read_csv("measurement_comparison.csv")
    df_comparison.drop("notes", axis=1, inplace=True)
    fig, ax = plt.subplots(figsize=(4, 2))
    sns.scatterplot(
        x="Percent N2",
        y="Cell Size (mm)",
        hue="Source",
        style="Initial Pressure (kPa)",
        data=df_comparison,
        palette=palette,
        ax=ax,
        linewidth=0,
        s=7.5,
    )
    plt.xlabel("Percent N$_2$")
    plt.legend(fontsize="x-small", markerscale=0.5)
    sns.despine()
    plt.tight_layout()
    file_name = f"akbar_comparison.{generate_images.PLOT_FILETYPE}"
    plt.savefig(
        os.path.join(
            generate_images.SAVE_LOC,
            file_name,
        ),
        dpi=generate_images.DPI,
    )
    plt.show()


if __name__ == "__main__":
    main()
