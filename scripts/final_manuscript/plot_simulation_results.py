import sqlite3
from dataclasses import dataclass
from sqlite3 import Connection
from typing import Literal, Optional, Union, Tuple

import matplotlib as mpl
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.ticker import FuncFormatter

from scripts.final_manuscript.plot_settings import set_palette, set_style


def load_conditions_data(con: Connection) -> pd.DataFrame:
    return pd.read_sql_query("select * from conditions", con)


def load_reactions_data(con: Connection) -> pd.DataFrame:
    return pd.read_sql_query(
        """
        select
            rc.condition_id,
            rc.run_no,
            rc.sim_type,
            rc.reaction,
            rc.diluent,
            rc.dil_mf,
            rc.time,
            rc.fwd_rate_constant,
            rc.fwd_rate_of_progress,
            rc.rev_rate_constant,
            rc.rev_rate_of_progress,
            rc.net_rate_of_progress,
            b.temperature,
            b.pressure,
            b.velocity
        from
            (
                (
            select
                *
            from
                reactions r
            inner join conditions c
                on
                c.id = r.condition_id
                ) rc
        inner join bulk_properties b
                on
            b.condition_id = rc.condition_id
            AND b.time = rc.time
            AND b.run_no = rc.run_no
           )
        """,
        con,
    )


def load_species_data(con: Connection) -> pd.DataFrame:
    return pd.read_sql_query(
        """
        select
            rc.condition_id,
            rc.run_no,
            rc.sim_type,
            rc.species,
            rc.diluent,
            rc.dil_mf,
            rc.time,
            rc.mole_frac,
            rc.concentration,
            rc.creation_rate,
            rc.destruction_rate,
            rc.net_production_rate,
            b.temperature,
            b.pressure,
            b.velocity
        from
            (
                (
            select
                *
            from
                species r
            inner join conditions c
                on
                c.id = r.condition_id
                ) rc
        inner join bulk_properties b
                on
            b.condition_id = rc.condition_id
            AND b.time = rc.time
            AND b.run_no = rc.run_no
           )
        """,
        con,
    )


@dataclass(frozen=True)
class DataColumn:
    column_name: str
    plot_name: str
    data_type: Union[Literal["reaction"], Literal["species"]]
    units: Optional[str]
    offset: Optional[float] = None


class Species:
    # Will have to generalize this a bit if we start using more complex species
    def __init__(self, element: str, number: int):
        self.element = element
        self.number = number

    def as_string(self) -> str:
        return f"{self.element}{self.number}"

    def as_tex_string(self) -> str:
        return f"{self.element}_{{{self.number}}}"


class DataColumnSpecies:
    mole_frac = DataColumn(
        column_name="mole_frac",
        plot_name="Mole Fraction",
        units="-",
        data_type="species",
        offset=1e-1,
    )
    concentration = DataColumn(
        column_name="concentration",
        plot_name="Concentration",
        units=r"\frac{ \mathrm{kmol} }{ \mathrm{m}^{3} }",
        data_type="species",
        offset=1e-2,
    )
    creation_rate = DataColumn(
        column_name="creation_rate",
        plot_name="Creation Rate",
        units=r"\frac{ \mathrm{kmol} }{ \mathrm{m}^{3} \cdot \mathrm{s} }",
        data_type="species",
        offset=1e8,
    )
    destruction_rate = DataColumn(
        column_name="destruction_rate",
        plot_name="Destruction Rate",
        units=r"\frac{ \mathrm{kmol} }{ \mathrm{m}^{3} \cdot \mathrm{s} }",
        data_type="species",
        offset=1e8,
    )
    net_production_rate = DataColumn(
        column_name="net_production_rate",
        plot_name="Net Production Rate",
        units=r"\frac{ \mathrm{kmol} }{ \mathrm{m}^{3} \cdot \mathrm{s} }",
        data_type="species",
        offset=1e8,
    )


class DataColumnReaction:
    fwd_rate_constant = DataColumn(
        column_name="fwd_rate_constant",
        plot_name="Forward Rate Constant",
        units=None,
        data_type="reaction",
        offset=1e9,
    )
    fwd_rate_of_progress = DataColumn(
        column_name="fwd_rate_of_progress",
        plot_name="Forward Rate of Progress",
        units=r"\frac{ \mathrm{kmol} }{ \mathrm{m}^{3} \cdot \mathrm{s} }",
        data_type="reaction",
        offset=1e6,
    )
    rev_rate_constant = DataColumn(
        column_name="rev_rate_constant",
        plot_name="Reverse Rate Constant",
        units=None,
        data_type="reaction",
        offset=1e9,
    )
    rev_rate_of_progress = DataColumn(
        column_name="rev_rate_of_progress",
        plot_name="Reverse Rate of Progress",
        units=r"\frac{ \mathrm{kmol} }{ \mathrm{m}^{3} \cdot \mathrm{s} }",
        data_type="reaction",
        offset=1e6,
    )
    net_rate_of_progress = DataColumn(
        column_name="net_rate_of_progress",
        plot_name="Net Rate of Progress",
        units=r"\frac{ \mathrm{kmol} }{ \mathrm{m}^{3} \cdot \mathrm{s} }",
        data_type="reaction",
        offset=1e6,
    )


class SimulationPlot:
    def __init__(
        self,
        velocity: Optional[plt.Axes],
        conditions: plt.Axes,
        results: plt.Axes,
        relative_dilution: Union[Literal["High"], Literal["Low"]],
        diluent: Optional[Species] = None,
        title_fontsize: int = 9,
        axis_fontsize: int = 6,
        legend_fontsize: int = 5,
    ):
        self.velocity = velocity
        self.temperature = conditions
        self.pressure = conditions.twinx()
        self.results = results
        self.relative_dilution = relative_dilution
        self.diluent = diluent

        self._dil_mf = None

        self._title_fontsize = title_fontsize
        self._axis_fontsize = axis_fontsize
        self._legend_fontsize = legend_fontsize

        all_axes = [self.temperature, self.pressure, self.results]
        if self.velocity is not None:
            all_axes.append(self.velocity)
        for axes in all_axes:
            axes.yaxis.get_offset_text().set_fontsize(self._axis_fontsize)
            axes.yaxis.set_major_formatter(FuncFormatter(lambda x, pos: f"{x:1.2f}"))
            for ax in (axes.xaxis, axes.yaxis):
                ax.set_tick_params(labelsize=self._axis_fontsize)
        self.set_title()

    def align_all_axes(self):
        self.pressure.yaxis.set_label_coords(1.125, 0.5)
        self.temperature.yaxis.set_label_coords(-0.125, 0.5)
        self.results.yaxis.set_label_coords(-0.125, 0.5)

    def set_dil_mf(self, dil_mf: float):
        self._dil_mf = dil_mf
        self.set_title()

    def set_title(self):
        title_axes = self.velocity or self.temperature
        title = f"{self.relative_dilution} dilution"
        if self._dil_mf is not None:
            title += r" $\left(\chi_" + f"{{{self.diluent.as_tex_string()}}} = {self._dil_mf:3.2f}" + r"\right)$"
        # we should be able to set this on temperature or pressure since they share a subplot
        title_axes.set_title(title, fontsize=self._title_fontsize)

    def plot_data(
        self,
        data: pd.DataFrame,
        data_column: DataColumn,
        result_designator_column: str,
        conditions: pd.Series,
        induction_window: Optional[float] = None,
    ):
        """
        Parameters
        ==========
        data: simulation data to plot
        data_column: column designating data to plot in the lower half of the combined plot
        result_designator_column : column designating reaction or species values, each of which will have a data plot
            entry
        conditions : series of current plot's test conditions and results
        induction_window : (Optional) time in seconds - plot x scale will be set to t_ind +/- `induction_window`
        """
        plot_data = data.sort_values("time")
        time_scale = 1_000_000  # s -> us
        temp_scale = 1e-3
        press_scale = 1e-7

        plot_data["time"] *= time_scale
        plot_data["pressure"] *= press_scale
        plot_data["temperature"] *= temp_scale

        dil_mfs = plot_data["dil_mf"].unique()
        n_dil_mfs = len(dil_mfs)
        if n_dil_mfs > 1:
            raise RuntimeError(f"Data does not have a unique `dil_mf`: {dil_mfs}")
        elif n_dil_mfs < 1:
            raise RuntimeError("Data does not have any `dil_mf`")
        self.set_dil_mf(dil_mfs[0])

        line_pressure = self.pressure.plot(plot_data["time"], plot_data["pressure"], ls="--", color="C1", label="P")
        line_temperature = self.temperature.plot(
            plot_data["time"], plot_data["temperature"], ls="-", color="C0", label="T"
        )

        t_min = 0
        t_max = plot_data["time"].max()
        if induction_window:
            t_ind = conditions["t_ind"]
            t_min = max((t_ind - induction_window) * time_scale, t_min)
            t_max = min((t_ind + induction_window) * time_scale, t_max)

        self.temperature.set_xlim(t_min, t_max)
        self.temperature.set_xlim(t_min, t_max)
        self.results.set_xlim(t_min, t_max)

        if self.velocity is not None:
            self.velocity.plot(plot_data["time"], plot_data["velocity"], "C5")
            self.velocity.set_xlim(t_min, t_max)
            self.velocity.set_ylabel("Velocity (m/s)", fontsize=self._axis_fontsize, verticalalignment="baseline")

        # Induction time is determined by a temperature threshold in the CV simulations, so that is where we plot it
        if conditions.sim_type == "cv":
            induction_time = conditions.t_ind * time_scale
            for ax in (self.temperature, self.results):
                ax.axvspan(0, induction_time, zorder=-1, color="#eee")

        self.pressure.set_ylabel(
            f"Pressure (Pa)\nx{press_scale}", fontsize=self._axis_fontsize, verticalalignment="top"
        )
        self.temperature.set_ylabel(
            f"Temperature (K)\nx{temp_scale:1.1e}", fontsize=self._axis_fontsize, verticalalignment="bottom"
        )

        lines = line_pressure + line_temperature
        labels = [line.get_label() for line in lines]
        # we should be able to set this on temperature or pressure since they share a subplot
        self.temperature.legend(lines, labels, loc=5, fontsize=self._legend_fontsize)

        # reaction/species plots
        color_indices = [2, 3, 4, 5, 6, 7]
        n_colors = len(color_indices)
        line_styles = ["-", "--", "-.", ":"]
        n_styles = len(line_styles)
        lines = []
        for i, (label, data_group) in enumerate(plot_data.groupby(result_designator_column)):
            color_idx = color_indices[i % n_colors]
            # if we expand the number of reactions/species and need more differentiation between lines, we can always
            # cycle through styles at a slower rate
            ls = line_styles[(i // n_colors) % n_styles]
            lines += self.results.plot(
                data_group["time"],
                data_group[data_column.column_name] / (data_column.offset or 1),
                color=f"C{color_idx}",
                label=label,
                ls=ls,
            )
        labels = [line.get_label() for line in lines]

        ylabel = data_column.plot_name
        if data_column.units:
            ylabel += (
                r" $\left("
                + f"{data_column.units}"
                + r"\right)$"
                + (f"\nx{1/data_column.offset:1.1e}" if data_column.offset else "")
            )
        self.results.legend(lines, labels, fontsize=self._legend_fontsize)
        self.results.set_ylabel(ylabel, fontsize=self._axis_fontsize)
        self.results.set_xlabel(r"Time $\left( \mu \mathrm{s} \right)$", fontsize=self._axis_fontsize)
        self.align_all_axes()


@dataclass
class SimulationPlotRow:
    dil_low: SimulationPlot
    dil_high: SimulationPlot
    row: plt.Axes


@dataclass(frozen=True)
class ResultConditions:
    dil_high: pd.Series
    dil_low: pd.Series


@dataclass(frozen=True)
class PlotConditions:
    sim_type: Union[Literal["cv"], Literal["znd"]]  # this can probably be smarter but whatever
    co2: ResultConditions
    n2_mf: ResultConditions
    n2_tad: ResultConditions


class SimulationPlots:
    def __init__(
        self,
        data: pd.DataFrame,
        data_column: DataColumn,
        condition_ids: PlotConditions,
        show_title: bool = False,
        save_to: Optional[str] = None,
        suptitle_fontsize: int = 12,
        row_fontsize: int = 10,
        ignore_middle: bool = False,
        induction_window: Optional[float] = None,
    ):
        # noinspection PyTypeChecker
        self._figure: plt.Figure = None
        self._suptitle_fontsize = suptitle_fontsize

        # these need to match self_rows in self._create_plots(), and are set using setattr()
        # noinspection PyTypeChecker
        self.co2: SimulationPlotRow = None  # set in _create_plots
        self.n2_mf: Optional[SimulationPlotRow] = None
        # noinspection PyTypeChecker
        self.n2_tad: SimulationPlotRow = None  # set in _create_plots

        plot_velocity = condition_ids.sim_type == "znd"

        self._create_plots(ignore_middle, plot_velocity)

        self.co2.row.set_title("CO$_{2}$ diluted", fontsize=row_fontsize)
        if not ignore_middle:
            self.n2_mf.row.set_title("N$_{2}$ diluted, mole fraction matched", fontsize=row_fontsize)
        self.n2_tad.row.set_title("N$_{2}$ diluted, T$_{ad}$ matched", fontsize=row_fontsize)

        if show_title:
            sim_kind = condition_ids.sim_type.upper()
            plot_kind = data_column.plot_name.replace("\n", " ")
            self._figure.suptitle(
                f"{sim_kind} Simulations ({plot_kind})",
                fontsize=self._suptitle_fontsize,
                y=0.925,
            )

        self._plot_data(data, data_column, condition_ids, ignore_middle, induction_window)

        self._share_ylim()

        self._figure.align_ylabels()

        if isinstance(save_to, str):
            self._figure.savefig(save_to)

    def _share_ylim(self):
        axes = []
        temp_bounds = [1e16, 0]
        press_bounds = [1e16, 0]
        result_bounds = [1e16, 0]
        vel_bounds = [1e16, 0]
        for row in (self.co2, self.n2_mf, self.n2_tad):
            if row is not None:
                for ax in (row.dil_low, row.dil_high):
                    axes.append(ax)

                    y_lim = ax.temperature.get_ylim()
                    temp_bounds[0] = min(temp_bounds[0], y_lim[0])
                    temp_bounds[1] = max(temp_bounds[1], y_lim[1])

                    y_lim = ax.pressure.get_ylim()
                    press_bounds[0] = min(press_bounds[0], y_lim[0])
                    press_bounds[1] = max(press_bounds[1], y_lim[1])

                    y_lim = ax.results.get_ylim()
                    result_bounds[0] = min(result_bounds[0], y_lim[0])
                    result_bounds[1] = max(result_bounds[1], y_lim[1])

                    if ax.velocity is not None:
                        y_lim = ax.velocity.get_ylim()
                        vel_bounds[0] = min(vel_bounds[0], y_lim[0])
                        vel_bounds[1] = max(vel_bounds[1], y_lim[1])

        for ax in axes:
            ax.temperature.set_ylim(*temp_bounds)
            ax.pressure.set_ylim(*press_bounds)
            ax.results.set_ylim(*result_bounds)

            if ax.velocity is not None:
                ax.velocity.set_ylim(*vel_bounds)

    def _create_plots(self, ignore_middle: bool, plot_velocity: bool):
        self._figure = plt.figure(figsize=(8.5, 11))

        n2 = Species("N", 2)
        co2 = Species("CO", 2)

        if ignore_middle:
            diluents = (co2, n2)
            self_rows = ("co2", "n2_tad")
        else:
            diluents = (co2, n2, n2)
            self_rows = ("co2", "n2_mf", "n2_tad")  # these need to match rows in __init__()
        rows_gridspec = self._figure.add_gridspec(nrows=len(diluents), ncols=1, hspace=0.3)
        for self_row, diluent, row_gs in zip(self_rows, diluents, rows_gridspec):
            row = self._figure.add_subplot(row_gs)
            row.axis("off")

            # This little guy keeps the high/low dilution titles from overlapping with the row titles
            row_plots_and_title_gs = row_gs.subgridspec(2, 1, height_ratios=[1, 100])

            n_windows_per_plot = 3 if plot_velocity else 2

            row_plots = row_plots_and_title_gs[1].subgridspec(n_windows_per_plot, 2, hspace=0, wspace=0.4)
            row_axes = []
            relative_dilutions = ["Low", "High"]
            for col, relative_dilution in enumerate(relative_dilutions):
                relative_dilution: Union[Literal["Low"], Literal["High"]]

                plot_idx = 0
                if plot_velocity:
                    ax_velocity = self._figure.add_subplot(row_plots[0, col])
                    ax_velocity.get_xaxis().set_visible(False)
                    plot_idx += 1
                else:
                    ax_velocity = None
                ax_conditions = self._figure.add_subplot(row_plots[plot_idx, col])
                ax_conditions.get_xaxis().set_visible(False)
                plot_idx += 1

                ax_results = self._figure.add_subplot(row_plots[plot_idx, col])
                row_axes.append(
                    SimulationPlot(
                        velocity=ax_velocity,
                        conditions=ax_conditions,
                        results=ax_results,
                        relative_dilution=relative_dilution,
                        diluent=diluent,
                    )
                )
            setattr(
                self,
                self_row,
                SimulationPlotRow(row=row, dil_low=row_axes[0], dil_high=row_axes[1]),
            )

    def _plot_data(
        self,
        data: pd.DataFrame,
        data_column: DataColumn,
        conditions: PlotConditions,
        ignore_middle: bool,
        induction_window: Optional[float],
    ):
        is_reaction_data = "reaction" in data.columns
        is_species_data = "species" in data.columns
        if is_reaction_data and is_species_data:
            raise RuntimeError("Data contains both reaction and species columns!")
        elif not (is_reaction_data or is_species_data):
            raise RuntimeError("Data doesn't contain reaction or species columns!")
        result_designator_column = "reaction" if is_reaction_data else "species"
        if data_column.data_type == "reaction" and is_species_data:
            raise RuntimeError("Reaction plots were requested but species data was provided!")
        elif data_column.data_type == "species" and is_reaction_data:
            raise RuntimeError("Species plots were requested but reaction data was provided!")

        # these need to match the keys for DataColumn and PlotConditionIds
        if ignore_middle:
            dilution_types = ("co2", "n2_tad")
        else:
            dilution_types = ("co2", "n2_mf", "n2_tad")
        relative_amounts = ("dil_low", "dil_high")
        for dilution_type in dilution_types:
            for relative_amount in relative_amounts:
                these_conditions = conditions.__dict__[dilution_type].__dict__[relative_amount]
                self.__dict__[dilution_type].__dict__[relative_amount].plot_data(
                    data=data[data["condition_id"] == these_conditions.id],
                    data_column=data_column,
                    result_designator_column=result_designator_column,
                    conditions=these_conditions,
                    induction_window=induction_window,
                )


def validate_sim_type(maybe: str) -> Union[Literal["cv"], Literal["znd"]]:
    if maybe == "cv":
        return "cv"
    elif maybe == "znd":
        return "znd"
    else:
        raise RuntimeError(f"Invalid simulation type: {maybe}")


def get_condition_ids(conditions: pd.DataFrame) -> tuple[PlotConditions, ...]:
    all_condition_ids = []
    for sim_type, sim_type_conditions in conditions.groupby("sim_type"):
        all_condition_ids.append(
            PlotConditions(
                sim_type=validate_sim_type(str(sim_type)),
                co2=ResultConditions(
                    dil_high=sim_type_conditions[
                        (
                            (sim_type_conditions["diluent"] == "CO2")
                            & (sim_type_conditions["match"].isna())
                            & (sim_type_conditions["dil_condition"] == "high")
                        )
                    ].iloc[0],
                    dil_low=sim_type_conditions[
                        (
                            (sim_type_conditions["diluent"] == "CO2")
                            & (sim_type_conditions["match"].isna())
                            & (sim_type_conditions["dil_condition"] == "low")
                        )
                    ].iloc[0],
                ),
                n2_mf=ResultConditions(
                    dil_high=sim_type_conditions[
                        (
                            (sim_type_conditions["diluent"] == "N2")
                            & (sim_type_conditions["match"] == "mf")
                            & (sim_type_conditions["dil_condition"] == "high")
                        )
                    ].iloc[0],
                    dil_low=sim_type_conditions[
                        (
                            (sim_type_conditions["diluent"] == "N2")
                            & (sim_type_conditions["match"] == "mf")
                            & (sim_type_conditions["dil_condition"] == "low")
                        )
                    ].iloc[0],
                ),
                n2_tad=ResultConditions(
                    dil_high=sim_type_conditions[
                        (
                            (sim_type_conditions["diluent"] == "N2")
                            & (sim_type_conditions["match"] == "tad")
                            & (sim_type_conditions["dil_condition"] == "high")
                        )
                    ].iloc[0],
                    dil_low=sim_type_conditions[
                        (
                            (sim_type_conditions["diluent"] == "N2")
                            & (sim_type_conditions["match"] == "tad")
                            & (sim_type_conditions["dil_condition"] == "low")
                        )
                    ].iloc[0],
                ),
            )
        )
    return tuple(all_condition_ids)


def main():
    set_palette()
    set_style()
    mpl.rcParams["lines.linewidth"] = 1

    show_plots = True
    save_plots = False
    show_title = True
    ignore_middle_plot = True
    induction_window = 1e-7

    db_path = "/home/mick/DetResearch/scripts/final_manuscript/co2_reaction_study.sqlite"
    con = sqlite3.connect(db_path)

    conditions = load_conditions_data(con)
    reactions = load_reactions_data(con)
    species = load_species_data(con)

    all_condition_ids = get_condition_ids(conditions)
    all_data_sources = (reactions, species)
    all_data_columns = (
        (
            DataColumnReaction.fwd_rate_of_progress,
            DataColumnReaction.fwd_rate_constant,
            DataColumnReaction.rev_rate_constant,
            DataColumnReaction.rev_rate_of_progress,
            DataColumnReaction.net_rate_of_progress,
        ),
        (
            DataColumnSpecies.concentration,
            DataColumnSpecies.creation_rate,
            DataColumnSpecies.mole_frac,
            DataColumnSpecies.destruction_rate,
            DataColumnSpecies.net_production_rate,
        ),
    )
    for data_source, data_columns in zip(all_data_sources, all_data_columns):
        for condition_ids in all_condition_ids:
            for data_column in data_columns:
                if save_plots:
                    save_to = (
                        f"plots/{condition_ids.sim_type} - {data_column.data_type} - {data_column.column_name}.pdf"
                    )
                else:
                    save_to = None
                SimulationPlots(
                    data=data_source,
                    condition_ids=condition_ids,
                    data_column=data_column,
                    show_title=show_title,
                    save_to=save_to,
                    ignore_middle=ignore_middle_plot,
                    induction_window=induction_window,
                )

    # zoomed in
    data_column = DataColumnReaction.fwd_rate_of_progress
    condition_ids = all_condition_ids[0]  # will break here if we add ZND back in
    SimulationPlots(
        data=reactions[~reactions.reaction.str.startswith("CO + OH")],
        condition_ids=condition_ids,
        data_column=data_column,
        show_title=show_title,
        save_to=f"plots/cv - {data_column.data_type}- {data_column.column_name} (zoomed).pdf",
        ignore_middle=ignore_middle_plot,
        induction_window=induction_window,
    )

    if show_plots:
        plt.show()


if __name__ == "__main__":
    main()
