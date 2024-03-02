import seaborn as sns


def set_palette():
    # IBM colorblind safe palette plus black and some extra colors manually checked for probably-goodness
    # https://davidmathlogic.com/colorblind/#%23648FFF-%23785EF0-%23DC267F-%23FE6100-%23FFB000-%E5B4FF-%2391ECFF
    colors = [
        # standard (rearranged a bit)
        "#648fff",  # temperature
        "#dc267f",  # pressure
        "#ffb000",  # rxn/spec
        "#785ef0",  # rxn/spec
        "#fe6100",  # rxn/spec
        # custom
        "#91ecff",  # rxn/spec
        "#e5b4ff",  # rxn/spec
        "#000000",  # rxn/spec
    ]
    sns.set_palette(colors)


def set_style(grid: bool = False):
    style = "white"
    if grid:
        style += "grid"
    sns.set_style(style)
