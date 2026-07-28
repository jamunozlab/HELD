#!/usr/bin/env python3
"""Animate every complete HELD phonon step from the Nose-Hoover trajectory."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-held-nose")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp")

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter


CASE_DIR = Path(__file__).resolve().parent
DEFAULT_CACHE = CASE_DIR / "held_heatmap_steps_all_valid.npz"
DEFAULT_STEPS = CASE_DIR / "held_steps_with_thermodynamics.csv"
DEFAULT_NPZ = CASE_DIR / "simulation_held.npz"
DEFAULT_OUTPUT = CASE_DIR / "bcc_Fe_nose_5000K_HELD_0001_0399.gif"
PATH_LABELS = [r"$\Gamma$", "H", "N", r"$\Gamma$", "P", "H"]
NAVY = "#0b2d4d"
ORANGE = "#f47c20"
BLUE = "#2878b5"
GREEN = "#2a9d68"
GRAY = "#6b7280"
LIGHT = "#d8dee7"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--steps-csv", type=Path, default=DEFAULT_STEPS)
    parser.add_argument("--npz", type=Path, default=DEFAULT_NPZ)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--fps", type=int, default=12)
    parser.add_argument("--dpi", type=int, default=100)
    return parser.parse_args()


def padded_limits(values: np.ndarray, fraction: float = 0.07) -> tuple[float, float]:
    finite = np.asarray(values, dtype=float)
    finite = finite[np.isfinite(finite)]
    low, high = float(np.min(finite)), float(np.max(finite))
    padding = fraction * (high - low) if high > low else 1.0
    return low - padding, high + padding


def main() -> int:
    args = parse_args()
    with np.load(args.cache, allow_pickle=False) as cache:
        x_values = np.asarray(cache["x_values"], dtype=float)
        step_frequencies = np.asarray(cache["step_frequencies_thz"], dtype=float)
    table = np.atleast_1d(np.genfromtxt(args.steps_csv, delimiter=",", names=True))
    if len(table) != len(step_frequencies):
        raise ValueError(f"{len(table)} CSV rows do not match {len(step_frequencies)} phonon frames")

    iteration = np.asarray(table["iteration"], dtype=float)
    time_ps = np.asarray(table["time_ps"], dtype=float)
    temperature = np.asarray(table["temperature_K"], dtype=float)
    pressure = np.asarray(table["pressure_GPa"], dtype=float)
    with np.load(args.npz, allow_pickle=False) as trajectory:
        valid = np.asarray(trajectory["frame_valid"], dtype=bool)
        energy = np.asarray(trajectory["energy_ry"], dtype=float)[valid]
    if len(energy) != len(step_frequencies):
        raise ValueError(f"{len(energy)} energy rows do not match {len(step_frequencies)} phonon frames")
    mean_frequencies = np.mean(step_frequencies, axis=0)

    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10.0})
    fig = plt.figure(figsize=(12.8, 7.2), facecolor="white")
    grid = fig.add_gridspec(
        3, 2, width_ratios=(2.05, 1.0), left=0.07, right=0.97,
        bottom=0.10, top=0.83, hspace=0.54, wspace=0.30,
    )
    phonon_axis = fig.add_subplot(grid[:, 0])
    observable_axes = [fig.add_subplot(grid[index, 1]) for index in range(3)]
    observable_axes[1].sharex(observable_axes[0])
    observable_axes[2].sharex(observable_axes[0])

    fig.suptitle(
        "BCC Iron at 5000 K: Step-Resolved HELD Phonons",
        x=0.07, y=0.965, ha="left", color=NAVY, fontsize=20, fontweight="bold",
    )
    fig.text(
        0.07, 0.915,
        "Nose-Hoover AIMD  |  128 atoms  |  4×4×4 BCC supercell  |  399 complete frames",
        ha="left", color=GRAY, fontsize=11,
    )
    status = fig.text(0.07, 0.865, "", ha="left", color=NAVY, fontsize=11.5, fontweight="bold")

    tick_indices = np.linspace(0, len(x_values) - 1, len(PATH_LABELS), dtype=int)
    tick_positions = x_values[tick_indices]
    for tick in tick_positions:
        phonon_axis.axvline(tick, color=LIGHT, linewidth=0.9, zorder=0)
    phonon_axis.axhline(0.0, color="#1f2937", linewidth=1.0, alpha=0.85)
    for branch_index in range(mean_frequencies.shape[1]):
        phonon_axis.plot(
            x_values, mean_frequencies[:, branch_index], color=NAVY,
            linewidth=1.1, alpha=0.42, linestyle="--",
            label="399-step mean" if branch_index == 0 else None,
        )
    branch_lines = [
        phonon_axis.plot(
            [], [], color=ORANGE, linewidth=2.3,
            label="Current step" if branch_index == 0 else None,
        )[0]
        for branch_index in range(step_frequencies.shape[2])
    ]
    phonon_axis.set_xlim(float(x_values[0]), float(x_values[-1]))
    phonon_axis.set_ylim(*padded_limits(step_frequencies, 0.05))
    phonon_axis.set_xticks(tick_positions)
    phonon_axis.set_xticklabels(PATH_LABELS, fontsize=12)
    phonon_axis.set_ylabel("Frequency (THz)", fontsize=12)
    phonon_axis.set_title("Phonon dispersion", loc="left", fontsize=13, fontweight="bold")
    phonon_axis.grid(axis="y", color=LIGHT, linewidth=0.7, alpha=0.65)
    phonon_axis.legend(loc="upper right", frameon=False)

    observable_series = (
        (temperature, ORANGE, "Temperature", "K"),
        (pressure, BLUE, "Pressure", "GPa"),
        (energy, GREEN, "Total energy", "Ry"),
    )
    history_lines = []
    markers = []
    for axis, (values, color, title, unit) in zip(observable_axes, observable_series):
        axis.plot(iteration, values, color=color, linewidth=1.0, alpha=0.20)
        history_lines.append(axis.plot([], [], color=color, linewidth=2.0)[0])
        markers.append(axis.plot([], [], "o", color=color, markersize=5.5, zorder=4)[0])
        axis.set_xlim(float(iteration[0]), float(iteration[-1]))
        axis.set_ylim(*padded_limits(values))
        axis.set_title(title, loc="left", fontsize=11.5, fontweight="bold")
        axis.set_ylabel(unit)
        axis.grid(color=LIGHT, linewidth=0.65, alpha=0.65)
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)
    observable_axes[-1].set_xlabel("MD iteration")
    plt.setp(observable_axes[0].get_xticklabels(), visible=False)
    plt.setp(observable_axes[1].get_xticklabels(), visible=False)

    def update(frame_index: int):
        current_slice = slice(0, frame_index + 1)
        for branch_index, line in enumerate(branch_lines):
            line.set_data(x_values, step_frequencies[frame_index, :, branch_index])
        for line, marker, (values, _color, _title, _unit) in zip(
            history_lines, markers, observable_series
        ):
            line.set_data(iteration[current_slice], values[current_slice])
            marker.set_data([iteration[frame_index]], [values[frame_index]])
        status.set_text(
            f"Step {int(iteration[frame_index]):03d}/{int(iteration[-1]):03d}"
            f"   •   t = {time_ps[frame_index]:.3f} ps"
            f"   •   T = {temperature[frame_index]:.0f} K"
            f"   •   P = {pressure[frame_index]:.2f} GPa"
            f"   •   E = {energy[frame_index]:.4f} Ry"
        )
        return (*branch_lines, *history_lines, *markers, status)

    animation = FuncAnimation(
        fig, update, frames=len(step_frequencies), interval=1000 / args.fps,
        blit=False, repeat=True,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    animation.save(
        args.output,
        writer=PillowWriter(
            fps=args.fps,
            metadata={
                "title": "BCC Fe Nose-Hoover 5000 K HELD phonons",
                "artist": "HELD",
            },
        ),
        dpi=args.dpi,
    )
    plt.close(fig)
    print(f"Saved {len(step_frequencies)} frames to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
