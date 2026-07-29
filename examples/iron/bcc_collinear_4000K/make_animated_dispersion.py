#!/usr/bin/env python3

from __future__ import annotations

import argparse
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-held")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp")

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter


CASE_DIR = Path(__file__).resolve().parent
DEFAULT_CACHE = CASE_DIR / "held_heatmap_steps_all_valid.npz"
DEFAULT_STEPS = CASE_DIR / "held_steps_with_thermodynamics.csv"
DEFAULT_OUTPUT = CASE_DIR / "bcc_collinear_4000K_HELD_0001_0112.gif"
PATH_LABELS = [r"$\Gamma$", "H", "N", r"$\Gamma$", "P", "H"]
COLORS = {
    "navy": "#0b2d4d",
    "orange": "#f47c20",
    "orange_dark": "#b84d00",
    "blue": "#2878b5",
    "green": "#2a9d68",
    "purple": "#7b4ab5",
    "gray": "#6b7280",
    "light_gray": "#d8dee7",
}


def padded_limits(values: np.ndarray, fraction: float = 0.08) -> tuple[float, float]:
    finite = np.asarray(values, dtype=float)
    finite = finite[np.isfinite(finite)]
    minimum = float(np.min(finite))
    maximum = float(np.max(finite))
    span = maximum - minimum
    padding = fraction * span if span > 0.0 else max(abs(maximum) * fraction, 1.0)
    return minimum - padding, maximum + padding


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Animate the step-resolved HELD dispersion and AIMD observables."
    )
    parser.add_argument("--cache", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--steps-csv", type=Path, default=DEFAULT_STEPS)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--fps", type=int, default=8)
    parser.add_argument("--dpi", type=int, default=120)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    with np.load(args.cache, allow_pickle=False) as cache:
        x_values = np.asarray(cache["x_values"], dtype=float)
        step_frequencies = np.asarray(cache["step_frequencies_thz"], dtype=float)

    table = np.genfromtxt(args.steps_csv, delimiter=",", names=True)
    table = np.atleast_1d(table)
    if len(table) != len(step_frequencies):
        raise ValueError(
            f"Thermodynamic rows ({len(table)}) do not match dispersion frames "
            f"({len(step_frequencies)})."
        )

    iteration = np.asarray(table["iteration"], dtype=float)
    time_ps = np.asarray(table["time_ps"], dtype=float)
    temperature = np.asarray(table["temperature_K"], dtype=float)
    pressure = np.asarray(table["pressure_GPa"], dtype=float)
    total_magnetization = np.asarray(table["mag_total_Bohr"], dtype=float)
    absolute_magnetization = np.asarray(table["abs_mag_total_Bohr"], dtype=float)
    mean_frequencies = np.nanmean(step_frequencies, axis=0)

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10.5,
            "axes.labelcolor": COLORS["navy"],
            "axes.titlecolor": COLORS["navy"],
            "xtick.color": "#334155",
            "ytick.color": "#334155",
        }
    )
    fig = plt.figure(figsize=(12.8, 7.2), facecolor="white")
    grid = fig.add_gridspec(
        3,
        2,
        width_ratios=(2.05, 1.0),
        left=0.07,
        right=0.97,
        bottom=0.10,
        top=0.83,
        hspace=0.54,
        wspace=0.30,
    )
    dispersion_axis = fig.add_subplot(grid[:, 0])
    temperature_axis = fig.add_subplot(grid[0, 1])
    pressure_axis = fig.add_subplot(grid[1, 1], sharex=temperature_axis)
    magnetization_axis = fig.add_subplot(grid[2, 1], sharex=temperature_axis)

    fig.suptitle(
        "BCC Iron at 4000 K: Step-Resolved HELD Phonons",
        x=0.07,
        y=0.965,
        ha="left",
        color=COLORS["navy"],
        fontsize=20,
        fontweight="bold",
    )
    fig.text(
        0.07,
        0.915,
        "Collinear-magnetic AIMD  |  128 atoms  |  ideal BCC symmetry reference",
        ha="left",
        color=COLORS["gray"],
        fontsize=11,
    )
    status_text = fig.text(
        0.07,
        0.865,
        "",
        ha="left",
        color=COLORS["navy"],
        fontsize=11.5,
        fontweight="bold",
    )

    tick_indices = np.linspace(0, len(x_values) - 1, len(PATH_LABELS), dtype=int)
    tick_positions = x_values[tick_indices]
    for tick_position in tick_positions:
        dispersion_axis.axvline(
            tick_position, color=COLORS["light_gray"], linewidth=0.9, zorder=0
        )
    dispersion_axis.axhline(0.0, color="#1f2937", linewidth=1.0, alpha=0.85)
    for branch_index in range(mean_frequencies.shape[1]):
        dispersion_axis.plot(
            x_values,
            mean_frequencies[:, branch_index],
            color=COLORS["navy"],
            linewidth=1.15,
            alpha=0.42,
            linestyle="--",
            label="112-step mean" if branch_index == 0 else None,
        )
    branch_lines = [
        dispersion_axis.plot(
            [],
            [],
            color=COLORS["orange"],
            linewidth=2.35,
            alpha=0.98,
            label="Current step" if branch_index == 0 else None,
        )[0]
        for branch_index in range(step_frequencies.shape[2])
    ]
    dispersion_axis.set_xlim(float(x_values[0]), float(x_values[-1]))
    dispersion_axis.set_ylim(*padded_limits(step_frequencies, fraction=0.06))
    dispersion_axis.set_xticks(tick_positions)
    dispersion_axis.set_xticklabels(PATH_LABELS, fontsize=12)
    dispersion_axis.set_ylabel("Frequency (THz)", fontsize=12)
    dispersion_axis.set_title("Phonon Dispersion", loc="left", fontsize=13, fontweight="bold")
    dispersion_axis.grid(axis="y", color=COLORS["light_gray"], linewidth=0.7, alpha=0.65)
    dispersion_axis.legend(loc="upper right", frameon=False, fontsize=10)

    observables = [
        (
            temperature_axis,
            temperature,
            COLORS["orange_dark"],
            "Temperature",
            "K",
        ),
        (pressure_axis, pressure, COLORS["blue"], "Pressure", "GPa"),
    ]
    history_lines = []
    current_markers = []
    for axis, values, color, title, unit in observables:
        axis.plot(iteration, values, color=color, linewidth=1.0, alpha=0.22)
        history_line = axis.plot([], [], color=color, linewidth=2.0)[0]
        marker = axis.plot([], [], "o", color=color, markersize=6, zorder=4)[0]
        history_lines.append(history_line)
        current_markers.append(marker)
        axis.set_xlim(float(iteration[0]), float(iteration[-1]))
        axis.set_ylim(*padded_limits(values))
        axis.set_title(title, loc="left", fontsize=11.5, fontweight="bold")
        axis.set_ylabel(unit)

    magnetization_axis.plot(
        iteration,
        total_magnetization,
        color=COLORS["purple"],
        linewidth=1.0,
        alpha=0.20,
    )
    magnetization_axis.plot(
        iteration,
        absolute_magnetization,
        color=COLORS["green"],
        linewidth=1.0,
        alpha=0.20,
    )
    total_history = magnetization_axis.plot(
        [], [], color=COLORS["purple"], linewidth=2.0, label=r"$M_{\mathrm{total}}$"
    )[0]
    absolute_history = magnetization_axis.plot(
        [], [], color=COLORS["green"], linewidth=2.0, label=r"$M_{\mathrm{absolute}}$"
    )[0]
    total_marker = magnetization_axis.plot(
        [], [], "o", color=COLORS["purple"], markersize=5, zorder=4
    )[0]
    absolute_marker = magnetization_axis.plot(
        [], [], "o", color=COLORS["green"], markersize=5, zorder=4
    )[0]
    magnetization_axis.set_xlim(float(iteration[0]), float(iteration[-1]))
    magnetization_axis.set_ylim(
        *padded_limits(np.concatenate([total_magnetization, absolute_magnetization]))
    )
    magnetization_axis.set_title(
        "Magnetization", loc="left", fontsize=11.5, fontweight="bold"
    )
    magnetization_axis.set_ylabel(r"$\mu_B$/cell")
    magnetization_axis.set_xlabel("MD iteration")
    magnetization_axis.legend(loc="best", frameon=False, fontsize=8.5, ncol=2)

    for axis in (temperature_axis, pressure_axis, magnetization_axis):
        axis.grid(color=COLORS["light_gray"], linewidth=0.65, alpha=0.65)
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)
    plt.setp(temperature_axis.get_xticklabels(), visible=False)
    plt.setp(pressure_axis.get_xticklabels(), visible=False)

    def update(frame_index: int):
        frame_slice = slice(0, frame_index + 1)
        for branch_index, branch_line in enumerate(branch_lines):
            branch_line.set_data(
                x_values, step_frequencies[frame_index, :, branch_index]
            )

        for history_line, marker, values in zip(
            history_lines,
            current_markers,
            (temperature, pressure),
        ):
            history_line.set_data(iteration[frame_slice], values[frame_slice])
            marker.set_data([iteration[frame_index]], [values[frame_index]])

        total_history.set_data(
            iteration[frame_slice], total_magnetization[frame_slice]
        )
        absolute_history.set_data(
            iteration[frame_slice], absolute_magnetization[frame_slice]
        )
        total_marker.set_data(
            [iteration[frame_index]], [total_magnetization[frame_index]]
        )
        absolute_marker.set_data(
            [iteration[frame_index]], [absolute_magnetization[frame_index]]
        )
        status_text.set_text(
            f"Step {int(iteration[frame_index]):03d}/{int(iteration[-1]):03d}"
            f"   •   t = {time_ps[frame_index]:.3f} ps"
            f"   •   T = {temperature[frame_index]:.0f} K"
            f"   •   P = {pressure[frame_index]:.2f} GPa"
            f"   •   M = {total_magnetization[frame_index]:.2f} "
            f"({absolute_magnetization[frame_index]:.2f} absolute) \u03bcB/cell"
        )
        return (
            *branch_lines,
            *history_lines,
            *current_markers,
            total_history,
            absolute_history,
            total_marker,
            absolute_marker,
            status_text,
        )

    animation = FuncAnimation(
        fig,
        update,
        frames=len(step_frequencies),
        interval=1000 / args.fps,
        blit=False,
        repeat=True,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    animation.save(
        args.output,
        writer=PillowWriter(
            fps=args.fps,
            metadata={
                "title": "BCC Fe collinear-magnetic HELD phonons at 4000 K",
                "artist": "IronCoreMD / HELD",
            },
        ),
        dpi=args.dpi,
    )
    plt.close(fig)
    print(f"Saved {len(step_frequencies)} frames to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
