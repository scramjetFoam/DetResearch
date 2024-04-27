from enum import Enum

import seaborn as sns


class PlotKind(Enum):
    REACTION = "reaction"
    CONDITION = "condition"


def set_palette(plot_kind: PlotKind):
    if plot_kind is PlotKind.REACTION:
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
    elif plot_kind is PlotKind.CONDITION:
        # consistent with frontiers paper
        colors = [
            "#fe6100",  # schlieren / CO2
            "#648fff",  # soot foil / N2
        ]
    else:
        raise ValueError("Invalid plot kind")
    sns.set_palette(colors)


def set_style(grid: bool = False):
    # todo: sizing
    style = "white"
    if grid:
        style += "grid"
    sns.set_style(style)
