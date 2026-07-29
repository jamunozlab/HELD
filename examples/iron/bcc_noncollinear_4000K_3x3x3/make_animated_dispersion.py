#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
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
DEFAULT_CACHE = CASE_DIR / "held_heatmap_steps_all_complete.npz"
DEFAULT_NPZ = CASE_DIR / "simulation_held_ideal_reference.npz"
DEFAULT_OUTPUT = CASE_DIR / "bcc_noncollinear_3x3x3_4000K_0003_0024.gif"
COLORS = {
    "navy": "#0b2d4d",
    "orange": "#c94f2d",
    "blue": "#2878b5",
    "gray": "#6b7280",
    "light": "#d7dce2",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Animate noncollinear BCC HELD phonons and structure.")
    parser.add_argument("--cache", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--npz", type=Path, default=DEFAULT_NPZ)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--fps", type=int, default=6)
    parser.add_argument("--dpi", type=int, default=180)
    return parser.parse_args()


def draw_cell(axis, cell: np.ndarray) -> None:
    origin = np.zeros(3)
    a, b, c = cell
    edges = [
        (origin, a), (origin, b), (origin, c), (a, a + b), (a, a + c),
        (b, a + b), (b, b + c), (c, a + c), (c, b + c),
        (a + b, a + b + c), (a + c, a + b + c), (b + c, a + b + c),
    ]
    for start, end in edges:
        axis.plot(*zip(start, end), color="#666666", linewidth=0.65, alpha=0.65)


def padded_limits(values: np.ndarray) -> tuple[float, float]:
    minimum = float(np.nanmin(values))
    maximum = float(np.nanmax(values))
    padding = 0.06 * max(maximum - minimum, 1.0)
    return min(0.0, minimum - padding), maximum + padding


def main() -> int:
    args = parse_args()
    with np.load(args.cache, allow_pickle=False) as cache:
        x_values = np.asarray(cache["x_values"], dtype=float)
        frequencies = np.asarray(cache["step_frequencies_thz"], dtype=float)
    with np.load(args.npz, allow_pickle=False) as data:
        valid = np.asarray(data["frame_valid"], dtype=bool)
        positions = np.asarray(data["positions"], dtype=float)[valid]
        ideal = np.asarray(data["initial_positions_alat"], dtype=float)
        cell = np.asarray(data["input_cell_parameters"], dtype=float)
        iteration = np.asarray(data["iteration"], dtype=int)[valid]
        time_ps = np.asarray(data["time_ps"], dtype=float)[valid]
        temperature = np.asarray(data["temperature_K"], dtype=float)[valid]
        pressure = np.asarray(data["pressure_GPa"], dtype=float)[valid]
        energy = np.asarray(data["energy_ry"], dtype=float)[valid]
        magnetization = np.asarray(data["mag_total_vector_Bohr"], dtype=float)[valid]
        absolute_magnetization = np.asarray(data["abs_mag_total_Bohr"], dtype=float)[valid]

    if len(frequencies) != len(positions):
        raise ValueError("HELD frequency frames do not match complete trajectory frames")

    mean_frequencies = np.mean(frequencies, axis=0)
    tick_indices = np.linspace(0, len(x_values) - 1, 6, dtype=int)
    tick_positions = x_values[tick_indices]
    tick_labels = [r"$\Gamma$", "H", "N", r"$\Gamma$", "P", "H"]
    figure = plt.figure(figsize=(14.0, 7.0), facecolor="white")
    dispersion_axis = figure.add_axes([0.07, 0.13, 0.88, 0.74])
    structure_axis = figure.add_axes([0.70, 0.14, 0.22, 0.31], projection="3d", facecolor="white")

    for tick in tick_positions:
        dispersion_axis.axvline(tick, color="#a8a8a8", linewidth=0.75, alpha=0.7)
    dispersion_axis.axhline(0.0, color="#333333", linewidth=0.8)
    for branch_index in range(mean_frequencies.shape[1]):
        dispersion_axis.plot(
            x_values, mean_frequencies[:, branch_index],
            color=COLORS["navy"], linewidth=1.0, linestyle="--", alpha=0.30,
            label=f"{len(frequencies)}-frame mean" if branch_index == 0 else None,
        )
    branch_lines = [
        dispersion_axis.plot([], [], color=COLORS["orange"], linewidth=2.0)[0]
        for _ in range(frequencies.shape[2])
    ]
    dispersion_axis.set_xlim(float(x_values[0]), float(x_values[-1]))
    dispersion_axis.set_ylim(*padded_limits(frequencies))
    dispersion_axis.set_xticks(tick_positions)
    dispersion_axis.set_xticklabels(tick_labels, fontsize=13)
    dispersion_axis.set_ylabel("Frequency (THz)", fontsize=13)
    dispersion_axis.grid(axis="y", color=COLORS["light"], linewidth=0.65, alpha=0.7)
    dispersion_axis.legend(loc="upper left", frameon=False)
    status = dispersion_axis.text(
        0.985, 0.98, "", transform=dispersion_axis.transAxes,
        ha="right", va="top", fontsize=11.2,
        bbox={"boxstyle": "round,pad=0.45", "facecolor": "white", "edgecolor": COLORS["light"], "alpha": 0.94},
    )
    figure.suptitle(
        "BCC Fe Noncollinear HELD Step Dispersion, 3×3×3 at 4000 K",
        fontsize=18, y=0.965, color=COLORS["navy"], fontweight="bold",
    )

    cell_length = float(np.max(np.linalg.norm(cell, axis=1)))
    ideal_cart = ideal @ cell

    def update(frame_index: int):
        for branch_index, line in enumerate(branch_lines):
            line.set_data(x_values, frequencies[frame_index, :, branch_index])

        structure_axis.clear()
        current_frac = np.mod(positions[frame_index], 1.0)
        current_cart = current_frac @ cell
        displacement_frac = current_frac - ideal
        displacement_frac -= np.rint(displacement_frac)
        displacement = displacement_frac @ cell
        draw_cell(structure_axis, cell)
        structure_axis.scatter(
            current_cart[:, 0], current_cart[:, 1], current_cart[:, 2],
            s=15, color="#e76f51", edgecolor="white", linewidth=0.25, depthshade=True,
        )
        structure_axis.quiver(
            ideal_cart[:, 0], ideal_cart[:, 1], ideal_cart[:, 2],
            displacement[:, 0], displacement[:, 1], displacement[:, 2],
            color=COLORS["blue"], linewidth=0.45, arrow_length_ratio=0.18, alpha=0.55,
        )
        vector = magnetization[frame_index]
        vector_norm = float(np.linalg.norm(vector))
        if vector_norm > 0.0:
            center = 0.5 * np.sum(cell, axis=0)
            direction = vector / vector_norm * 0.28 * cell_length
            structure_axis.quiver(
                *center, *direction, color=COLORS["navy"], linewidth=2.2,
                arrow_length_ratio=0.20,
            )
        structure_axis.set_xlim(0.0, cell_length)
        structure_axis.set_ylim(0.0, cell_length)
        structure_axis.set_zlim(0.0, cell_length)
        structure_axis.set_box_aspect((1, 1, 1))
        structure_axis.view_init(elev=19, azim=-58)
        structure_axis.set_axis_off()
        structure_axis.set_title("Atoms, displacements, and net M", fontsize=9.5, pad=2)

        status.set_text(
            f"Frame {frame_index + 1}/{len(frequencies)}  |  MD step {iteration[frame_index]}\n"
            f"t = {time_ps[frame_index]:.3f} ps   T = {temperature[frame_index]:.0f} K\n"
            f"P = {pressure[frame_index]:.1f} GPa   E = {energy[frame_index]:.3f} Ry\n"
            f"M = ({vector[0]:.2f}, {vector[1]:.2f}, {vector[2]:.2f}) μB\n"
            f"|M| = {vector_norm:.2f} μB   Mabs = {absolute_magnetization[frame_index]:.2f} μB"
        )
        return *branch_lines, status

    animation = FuncAnimation(
        figure, update, frames=len(frequencies), interval=1000 / args.fps,
        blit=False, repeat=True,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    animation.save(
        args.output,
        writer=PillowWriter(
            fps=args.fps,
            metadata={
                "title": "BCC Fe noncollinear 3x3x3 HELD phonons",
                "artist": "IronCoreMD / HELD",
            },
        ),
        dpi=args.dpi,
    )
    plt.close(figure)
    print(f"Saved {len(frequencies)} frames to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
