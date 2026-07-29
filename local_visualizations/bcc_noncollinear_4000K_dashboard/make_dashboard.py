#!/usr/bin/env python3
"""Render an isolated noncollinear BCC AIMD dashboard and GIF."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-held-dashboard")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp")

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.colors import Normalize


HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[2]
DEFAULT_NPZ = REPO_ROOT / "dataset/bcc/magnetic-non_coll/simulation.npz"
DEFAULT_HELD_CACHE = REPO_ROOT / "HELD/examples/iron/bcc_noncollinear_4000K_3x3x3/held_heatmap_steps_all_complete.npz"
DEFAULT_PNG = HERE / "bcc_noncollinear_md_dashboard_final.png"
DEFAULT_GIF = HERE / "bcc_noncollinear_md_dashboard.gif"

COLORS = {
    "navy": "#0b2d4d",
    "orange": "#f47c20",
    "purple": "#7b4ab5",
    "green": "#2a9d68",
    "blue": "#2878b5",
    "red": "#c44e52",
    "gray": "#687386",
    "light": "#d8dee7",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--npz", type=Path, default=DEFAULT_NPZ)
    parser.add_argument("--held-cache", type=Path, default=DEFAULT_HELD_CACHE)
    parser.add_argument("--png", type=Path, default=DEFAULT_PNG)
    parser.add_argument("--gif", type=Path, default=DEFAULT_GIF)
    parser.add_argument("--fps", type=int, default=5)
    parser.add_argument("--dpi", type=int, default=150)
    return parser.parse_args()


def metadata(data: np.lib.npyio.NpzFile) -> dict[str, object]:
    return json.loads(str(data["metadata_json"].item()))


def bcc_ideal_fractional(natoms: int) -> np.ndarray:
    repetitions = round((natoms / 2.0) ** (1.0 / 3.0))
    if 2 * repetitions**3 != natoms:
        raise ValueError(f"Expected a conventional BCC supercell, received {natoms} atoms")
    corners = []
    centers = []
    for z_index in range(repetitions):
        for y_index in range(repetitions):
            for x_index in range(repetitions):
                corners.append([x_index, y_index, z_index])
                centers.append([x_index + 0.5, y_index + 0.5, z_index + 0.5])
    return np.asarray(corners + centers, dtype=float) / repetitions


def cell_angstrom(data: np.lib.npyio.NpzFile, meta: dict[str, object]) -> np.ndarray:
    cell = np.asarray(data["input_cell_parameters"], dtype=float)
    unit = str(np.asarray(data["input_cell_unit"]).item()).lower()
    if np.isfinite(cell).all() and unit in {"angstrom", "ang"}:
        return cell
    return np.asarray(data["initial_cell_alat"], dtype=float) * float(meta["alat_bohr"]) * 0.529177210903


def displacement_cartesian(
    positions_fractional: np.ndarray, reference_fractional: np.ndarray, cell: np.ndarray
) -> np.ndarray:
    delta = positions_fractional - reference_fractional
    delta -= np.rint(delta)
    return delta @ cell


def draw_cell(axis, cell: np.ndarray) -> None:
    origin = np.zeros(3)
    a_vector, b_vector, c_vector = cell
    edges = [
        (origin, a_vector), (origin, b_vector), (origin, c_vector),
        (a_vector, a_vector + b_vector), (a_vector, a_vector + c_vector),
        (b_vector, a_vector + b_vector), (b_vector, b_vector + c_vector),
        (c_vector, a_vector + c_vector), (c_vector, b_vector + c_vector),
        (a_vector + b_vector, a_vector + b_vector + c_vector),
        (a_vector + c_vector, a_vector + b_vector + c_vector),
        (b_vector + c_vector, a_vector + b_vector + c_vector),
    ]
    for start, end in edges:
        axis.plot(*zip(start, end), color="#64748b", linewidth=1.05, alpha=0.75)


def style_history_axis(axis) -> None:
    axis.grid(color=COLORS["light"], linewidth=0.8, alpha=0.8)
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    axis.tick_params(labelsize=10)


def main() -> int:
    args = parse_args()
    with np.load(args.npz, allow_pickle=False) as data:
        meta = metadata(data)
        valid = np.asarray(data["frame_valid"], dtype=bool)
        valid &= np.asarray(data["local_magnetization_frame_valid"], dtype=bool)
        positions = np.asarray(data["positions"], dtype=float)[valid]
        iterations = np.asarray(data["iteration"], dtype=int)[valid]
        time_ps = np.asarray(data["time_ps"], dtype=float)[valid]
        temperature = np.asarray(data["temperature_K"], dtype=float)[valid]
        pressure = np.asarray(data["pressure_GPa"], dtype=float)[valid]
        magnetization = np.asarray(data["mag_total_vector_Bohr"], dtype=float)[valid]
        local_magnetization = np.asarray(data["local_magnetization_Bohr"], dtype=float)[valid]
        absolute_magnetization = np.asarray(data["abs_mag_total_Bohr"], dtype=float)[valid]
        cell = cell_angstrom(data, meta)
    with np.load(args.held_cache, allow_pickle=False) as held_cache:
        phonon_path = np.asarray(held_cache["x_values"], dtype=float)
        phonon_frequencies = np.asarray(held_cache["step_frequencies_thz"], dtype=float)
    if len(phonon_frequencies) != len(positions):
        raise ValueError(
            f"HELD frames {len(phonon_frequencies)} do not match trajectory frames {len(positions)}"
        )

    ideal = bcc_ideal_fractional(positions.shape[1])
    ideal_cartesian = ideal @ cell
    displacements = np.asarray(
        [displacement_cartesian(frame, ideal, cell) for frame in positions]
    )
    displacement_from_start = np.asarray(
        [displacement_cartesian(frame, positions[0], cell) for frame in positions]
    )
    displacement_magnitude = np.linalg.norm(displacements, axis=2)
    rms_ideal = np.sqrt(np.mean(displacement_magnitude**2, axis=1))
    rms_start = np.sqrt(np.mean(np.sum(displacement_from_start**2, axis=2), axis=1))
    maximum_ideal = np.max(displacement_magnitude, axis=1)
    magnetic_magnitude = np.linalg.norm(magnetization, axis=1)
    local_magnetic_magnitude = np.linalg.norm(local_magnetization, axis=2)
    maximum_local_moment = float(np.max(local_magnetic_magnitude))

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "axes.titleweight": "bold",
            "axes.titlecolor": COLORS["navy"],
            "axes.labelcolor": COLORS["navy"],
        }
    )
    figure = plt.figure(figsize=(17, 10), facecolor="white")
    grid = figure.add_gridspec(
        4, 2, width_ratios=(1.55, 1.0), height_ratios=(0.9, 0.8, 0.8, 1.15),
        left=0.045, right=0.975, bottom=0.075, top=0.84,
        wspace=0.18, hspace=0.48,
    )
    structure_axis = figure.add_subplot(grid[:, 0], projection="3d")
    magnetic_axis = figure.add_subplot(grid[0, 1])
    temperature_axis = figure.add_subplot(grid[1, 1])
    displacement_axis = figure.add_subplot(grid[2, 1])
    phonon_axis = figure.add_subplot(grid[3, 1])

    component_lines = []
    for component, color, label in zip(
        magnetization.T,
        (COLORS["purple"], COLORS["green"], COLORS["blue"]),
        (r"$M_x$", r"$M_y$", r"$M_z$"),
    ):
        component_lines.append(
            magnetic_axis.plot(iterations, component, color=color, linewidth=2.3, label=label)[0]
        )
    magnetic_axis.plot(
        iterations, magnetic_magnitude, color=COLORS["navy"], linewidth=1.8,
        linestyle="--", label=r"$|\mathbf{M}|$",
    )
    magnetic_axis.axhline(0.0, color="#334155", linewidth=0.9)
    magnetic_axis.set_title("Global Magnetization Components", loc="left", fontsize=14)
    magnetic_axis.set_ylabel(r"Magnetization ($\mu_B$/cell)", fontsize=11)
    magnetic_axis.legend(frameon=False, ncol=4, fontsize=10, loc="upper center")

    temperature_axis.plot(
        iterations, temperature, color=COLORS["red"], linewidth=2.5,
    )
    temperature_axis.axhline(
        np.mean(temperature), color=COLORS["red"], linestyle="--", alpha=0.45,
    )
    temperature_axis.set_title("Ionic Temperature", loc="left", fontsize=14)
    temperature_axis.set_ylabel("Temperature (K)", fontsize=11)

    displacement_axis.plot(
        iterations, rms_ideal, color=COLORS["orange"], linewidth=2.5,
        label="RMS from ideal BCC",
    )
    displacement_axis.plot(
        iterations, maximum_ideal, color=COLORS["red"], linewidth=2.0,
        label="Maximum from ideal BCC",
    )
    displacement_axis.plot(
        iterations, rms_start, color=COLORS["blue"], linewidth=2.2,
        label="RMS motion from first frame",
    )
    displacement_axis.set_title("Displacement History", loc="left", fontsize=14)
    displacement_axis.set_xlabel("MD iteration", fontsize=11)
    displacement_axis.set_ylabel("Displacement (Å)", fontsize=11)
    displacement_axis.legend(frameon=False, fontsize=9.5, loc="upper center")

    mean_phonons = np.mean(phonon_frequencies, axis=0)
    phonon_ticks = np.linspace(0, len(phonon_path) - 1, 6, dtype=int)
    phonon_tick_positions = phonon_path[phonon_ticks]
    for tick_position in phonon_tick_positions:
        phonon_axis.axvline(tick_position, color="#a8a8a8", linewidth=0.7, alpha=0.65)
    phonon_axis.axhline(0.0, color="#334155", linewidth=0.8)
    for branch_index in range(mean_phonons.shape[1]):
        phonon_axis.plot(
            phonon_path, mean_phonons[:, branch_index], color=COLORS["navy"],
            linewidth=1.0, linestyle="--", alpha=0.30,
            label=f"{len(phonon_frequencies)}-frame mean" if branch_index == 0 else None,
        )
    phonon_lines = [
        phonon_axis.plot([], [], color=COLORS["orange"], linewidth=2.0)[0]
        for _ in range(phonon_frequencies.shape[2])
    ]
    phonon_minimum = float(np.min(phonon_frequencies))
    phonon_maximum = float(np.max(phonon_frequencies))
    phonon_padding = 0.07 * max(phonon_maximum - phonon_minimum, 1.0)
    phonon_axis.set_xlim(float(phonon_path[0]), float(phonon_path[-1]))
    phonon_axis.set_ylim(phonon_minimum - phonon_padding, phonon_maximum + phonon_padding)
    phonon_axis.set_xticks(phonon_tick_positions)
    phonon_axis.set_xticklabels([r"$\Gamma$", "H", "N", r"$\Gamma$", "P", "H"])
    phonon_axis.set_title("HELD Step Phonons", loc="left", fontsize=14)
    phonon_axis.set_ylabel("Frequency (THz)", fontsize=11)
    phonon_axis.legend(frameon=False, fontsize=9, loc="upper left")

    for axis in (magnetic_axis, temperature_axis, displacement_axis, phonon_axis):
        style_history_axis(axis)
    for axis in (magnetic_axis, temperature_axis, displacement_axis):
        axis.set_xlim(iterations[0] - 0.5, iterations[-1] + 0.5)

    magnetic_cursor = magnetic_axis.axvline(iterations[0], color=COLORS["orange"], linewidth=1.4)
    temperature_cursor = temperature_axis.axvline(iterations[0], color=COLORS["orange"], linewidth=1.4)
    displacement_cursor = displacement_axis.axvline(iterations[0], color=COLORS["orange"], linewidth=1.4)
    magnetic_markers = [
        magnetic_axis.plot([], [], marker="o", markersize=6, color=line.get_color())[0]
        for line in component_lines
    ]
    temperature_marker = temperature_axis.plot([], [], "o", color=COLORS["red"], markersize=7)[0]
    displacement_markers = [
        displacement_axis.plot([], [], "o", color=color, markersize=6)[0]
        for color in (COLORS["orange"], COLORS["red"], COLORS["blue"])
    ]

    normalization = Normalize(vmin=-1.0, vmax=1.0)
    color_map = plt.get_cmap("coolwarm")
    color_bar_axis = figure.add_axes([0.11, 0.085, 0.34, 0.022])
    scalar_map = plt.cm.ScalarMappable(norm=normalization, cmap=color_map)
    color_bar = figure.colorbar(scalar_map, cax=color_bar_axis, orientation="horizontal")
    color_bar.set_label(r"Evolved local-spin direction, $m_z/|\mathbf{m}|$", fontsize=10)
    color_bar.ax.tick_params(labelsize=9)

    status = figure.text(
        0.075, 0.82, "", ha="left", va="top", fontsize=12.2, color=COLORS["navy"],
        bbox={"boxstyle": "round,pad=0.55", "facecolor": "white", "edgecolor": COLORS["light"], "alpha": 0.94},
    )
    figure.suptitle(
        "Noncollinear BCC Fe AIMD Dashboard",
        x=0.045, y=0.975, ha="left", fontsize=25, fontweight="bold", color=COLORS["navy"],
    )
    figure.text(
        0.047, 0.925,
        "Moving atoms with evolving QE local moments and synchronized histories",
        fontsize=12.5, color=COLORS["gray"],
    )

    cell_length = float(np.max(np.linalg.norm(cell, axis=1)))

    def update(frame_index: int):
        structure_axis.clear()
        current_fractional = np.mod(positions[frame_index], 1.0)
        current_cartesian = current_fractional @ cell

        draw_cell(structure_axis, cell)
        structure_axis.scatter(
            ideal_cartesian[:, 0], ideal_cartesian[:, 1], ideal_cartesian[:, 2],
            s=25, facecolor="none", edgecolor="#94a3b8", linewidth=0.7, alpha=0.55,
        )
        structure_axis.scatter(
            current_cartesian[:, 0], current_cartesian[:, 1], current_cartesian[:, 2],
            s=72, color=COLORS["orange"],
            edgecolor="white", linewidth=0.6, depthshade=True,
        )
        current_local_magnetization = local_magnetization[frame_index]
        current_local_norm = local_magnetic_magnitude[frame_index]
        current_local_direction_z = np.divide(
            current_local_magnetization[:, 2], current_local_norm,
            out=np.zeros_like(current_local_norm), where=current_local_norm > 0.0,
        )
        displayed_spins = 0.72 * current_local_magnetization / maximum_local_moment
        structure_axis.quiver(
            current_cartesian[:, 0], current_cartesian[:, 1], current_cartesian[:, 2],
            displayed_spins[:, 0], displayed_spins[:, 1], displayed_spins[:, 2],
            color=color_map(normalization(current_local_direction_z)), linewidth=1.45,
            arrow_length_ratio=0.30, alpha=0.95,
        )
        vector = magnetization[frame_index]
        vector_norm = magnetic_magnitude[frame_index]
        if vector_norm > 0.0:
            center = 0.5 * np.sum(cell, axis=0)
            displayed_global_m = 1.15 * vector / vector_norm
            structure_axis.quiver(
                *center, *displayed_global_m, color=COLORS["navy"], linewidth=4.0,
                arrow_length_ratio=0.22,
            )
            structure_axis.text(
                *(center + displayed_global_m), r" global $\mathbf{M}$",
                color=COLORS["navy"], fontsize=10, fontweight="bold",
            )
        structure_axis.set_xlim(-0.15, cell_length + 0.15)
        structure_axis.set_ylim(-0.15, cell_length + 0.15)
        structure_axis.set_zlim(-0.15, cell_length + 0.15)
        structure_axis.set_box_aspect((1, 1, 1))
        structure_axis.view_init(elev=21, azim=-57)
        structure_axis.set_xlabel("x (Å)", labelpad=9)
        structure_axis.set_ylabel("y (Å)", labelpad=9)
        structure_axis.set_zlabel("z (Å)", labelpad=9)
        structure_axis.set_title(
            f"Atomic Motion and Spin Texture — MD Iteration {iterations[frame_index]}",
            loc="left", fontsize=16, pad=18,
        )

        iteration = iterations[frame_index]
        for cursor in (magnetic_cursor, temperature_cursor, displacement_cursor):
            cursor.set_xdata([iteration, iteration])
        for component_index, marker in enumerate(magnetic_markers):
            marker.set_data([iteration], [magnetization[frame_index, component_index]])
        temperature_marker.set_data([iteration], [temperature[frame_index]])
        for marker, value in zip(
            displacement_markers,
            (rms_ideal[frame_index], maximum_ideal[frame_index], rms_start[frame_index]),
        ):
            marker.set_data([iteration], [value])
        for branch_index, branch_line in enumerate(phonon_lines):
            branch_line.set_data(
                phonon_path, phonon_frequencies[frame_index, :, branch_index]
            )

        status.set_text(
            f"Frame {frame_index + 1}/{len(iterations)}   •   t = {time_ps[frame_index]:.3f} ps\n"
            f"T = {temperature[frame_index]:.0f} K   •   P = {pressure[frame_index]:.1f} GPa\n"
            f"M = ({vector[0]:+.2f}, {vector[1]:+.2f}, {vector[2]:+.2f}) μB\n"
            f"|M| = {magnetic_magnitude[frame_index]:.2f} μB   •   Mabs = {absolute_magnetization[frame_index]:.2f} μB\n"
            f"RMS displacement = {rms_ideal[frame_index]:.3f} Å\n"
            f"Local moments: mean = {np.mean(current_local_norm):.2f} μB   •   max = {np.max(current_local_norm):.2f} μB"
        )
        return (
            magnetic_cursor, temperature_cursor, displacement_cursor,
            *magnetic_markers, temperature_marker, *displacement_markers,
            *phonon_lines, status,
        )

    args.png.parent.mkdir(parents=True, exist_ok=True)
    update(len(iterations) - 1)
    figure.savefig(args.png, dpi=220, facecolor="white")

    animation = FuncAnimation(
        figure, update, frames=len(iterations), interval=1000 / args.fps,
        blit=False, repeat=True,
    )
    animation.save(
        args.gif,
        writer=PillowWriter(
            fps=args.fps,
            metadata={"title": "Noncollinear BCC Fe AIMD dashboard", "artist": "IronCoreMD / HELD"},
        ),
        dpi=args.dpi,
    )
    plt.close(figure)
    print(f"static_dashboard={args.png}")
    print(f"animated_dashboard={args.gif}")
    print(f"frames={len(iterations)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
