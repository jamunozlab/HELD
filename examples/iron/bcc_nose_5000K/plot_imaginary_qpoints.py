#!/usr/bin/env python3
"""Plot the q-resolved imaginary modes in the step-resolved HELD cache."""

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
from matplotlib.colors import LinearSegmentedColormap


CASE_DIR = Path(__file__).resolve().parent
DEFAULT_CACHE = CASE_DIR / "held_heatmap_steps_all_valid.npz"
DEFAULT_STEPS = CASE_DIR / "held_steps_with_thermodynamics.csv"
DEFAULT_PLOT = CASE_DIR / "held_imaginary_qpoints.png"
DEFAULT_CSV = CASE_DIR / "held_imaginary_qpoints.csv"
PATH_LABELS = [r"$\Gamma$", "H", "N", r"$\Gamma$", "P", "H"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--steps-csv", type=Path, default=DEFAULT_STEPS)
    parser.add_argument("--output", type=Path, default=DEFAULT_PLOT)
    parser.add_argument("--output-csv", type=Path, default=DEFAULT_CSV)
    parser.add_argument(
        "--threshold-thz",
        type=float,
        default=0.01,
        help="Count frequencies below minus this positive threshold (default: 0.01 THz).",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.threshold_thz <= 0:
        raise ValueError("--threshold-thz must be positive")

    with np.load(args.cache, allow_pickle=False) as cache:
        x_values = np.asarray(cache["x_values"], dtype=float)
        frequencies = np.asarray(cache["step_frequencies_thz"], dtype=float)
    table = np.atleast_1d(np.genfromtxt(args.steps_csv, delimiter=",", names=True))
    iteration = np.asarray(table["iteration"], dtype=int)
    if len(iteration) != len(frequencies):
        raise ValueError("Step table and frequency-cache frame counts do not match")

    minimum_branch = np.min(frequencies, axis=2)
    imaginary = minimum_branch < -args.threshold_thz
    negative_count = np.count_nonzero(imaginary, axis=0)
    negative_fraction = negative_count / len(iteration)
    deepest_frequency = np.min(minimum_branch, axis=0)
    deepest_imaginary = np.where(negative_count > 0, deepest_frequency, np.nan)
    conditional_mean = np.full(len(x_values), np.nan, dtype=float)
    for q_index in range(len(x_values)):
        values = minimum_branch[imaginary[:, q_index], q_index]
        if len(values):
            conditional_mean[q_index] = float(np.mean(values))

    tick_indices = np.linspace(0, len(x_values) - 1, len(PATH_LABELS), dtype=int)
    tick_positions = x_values[tick_indices]
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    np.savetxt(
        args.output_csv,
        np.column_stack(
            [
                np.arange(len(x_values)),
                x_values,
                negative_count,
                negative_fraction,
                conditional_mean,
                deepest_frequency,
            ]
        ),
        delimiter=",",
        header=(
            "q_index,x_path,negative_frame_count,negative_frame_fraction,"
            "mean_negative_frequency_THz,minimum_frequency_THz"
        ),
        comments="",
        fmt=["%d", "%.10f", "%d", "%.10f", "%.10f", "%.10f"],
    )

    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10.5})
    figure = plt.figure(figsize=(12.0, 10.0), facecolor="white")
    grid = figure.add_gridspec(
        3, 1, height_ratios=(1.0, 1.0, 1.65),
        left=0.10, right=0.96, bottom=0.08, top=0.88, hspace=0.18,
    )
    occurrence_axis = figure.add_subplot(grid[0])
    depth_axis = figure.add_subplot(grid[1], sharex=occurrence_axis)
    timeline_axis = figure.add_subplot(grid[2], sharex=occurrence_axis)

    figure.suptitle(
        "BCC Fe Nose-Hoover AIMD: Instantaneous Imaginary HELD Modes",
        x=0.10, y=0.965, ha="left", color="#0b2d4d",
        fontsize=18, fontweight="bold",
    )
    figure.text(
        0.10, 0.925,
        (
            f"399 fitted frames  |  imaginary threshold: frequency < "
            f"−{args.threshold_thz:.2f} THz  |  mean-coefficient dispersion remains stable"
        ),
        ha="left", color="#5f6b7a", fontsize=10.5,
    )

    occurrence_axis.fill_between(
        x_values, 100.0 * negative_fraction,
        color="#d9483b", alpha=0.30, linewidth=0,
    )
    occurrence_axis.plot(
        x_values, 100.0 * negative_fraction,
        color="#b42318", linewidth=1.6,
    )
    occurrence_axis.set_ylabel("Frames with an\nimaginary mode (%)")
    occurrence_axis.set_ylim(
        0.0, max(55.0, 1.05 * float(np.max(100.0 * negative_fraction)))
    )
    occurrence_axis.grid(axis="y", color="#d8dee7", alpha=0.65)

    depth_axis.axhline(0.0, color="#263238", linewidth=0.8)
    depth_axis.fill_between(
        x_values, deepest_imaginary, 0.0,
        where=np.isfinite(deepest_imaginary),
        color="#7f1d1d", alpha=0.16,
    )
    depth_axis.plot(
        x_values, deepest_imaginary,
        color="#7f1d1d", linewidth=1.6, label="Deepest frame",
    )
    depth_axis.plot(
        x_values, conditional_mean,
        color="#f47c20", linewidth=1.5, label="Mean when imaginary",
    )
    depth_axis.set_ylabel("Signed frequency\n(THz)")
    depth_axis.legend(loc="lower left", frameon=False, ncol=2)
    depth_axis.grid(axis="y", color="#d8dee7", alpha=0.65)

    imaginary_map = np.ma.masked_where(~imaginary, minimum_branch)
    colormap = LinearSegmentedColormap.from_list(
        "imaginary_modes", ["#3b0710", "#9f1239", "#ef4444", "#fca5a5"]
    )
    image = timeline_axis.imshow(
        imaginary_map,
        origin="lower",
        aspect="auto",
        cmap=colormap,
        vmin=float(np.min(minimum_branch)),
        vmax=-args.threshold_thz,
        extent=(x_values[0], x_values[-1], iteration[0], iteration[-1]),
        interpolation="nearest",
    )
    timeline_axis.set_facecolor("#f4f6f8")
    timeline_axis.set_ylabel("MD iteration")
    timeline_axis.set_xlabel("BCC symmetry path")
    colorbar = figure.colorbar(image, ax=timeline_axis, pad=0.015, fraction=0.025)
    colorbar.set_label("Most negative branch (THz)")

    for axis in (occurrence_axis, depth_axis, timeline_axis):
        for tick in tick_positions:
            axis.axvline(tick, color="#87909c", linewidth=0.7, alpha=0.55)
        axis.set_xlim(float(x_values[0]), float(x_values[-1]))
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)
    timeline_axis.set_xticks(tick_positions)
    timeline_axis.set_xticklabels(PATH_LABELS, fontsize=12)
    plt.setp(occurrence_axis.get_xticklabels(), visible=False)
    plt.setp(depth_axis.get_xticklabels(), visible=False)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.output, dpi=220)
    plt.close(figure)

    unstable_frames = int(np.count_nonzero(np.any(imaginary, axis=1)))
    unstable_qpoints = int(np.count_nonzero(np.any(imaginary, axis=0)))
    minimum_index = np.unravel_index(np.argmin(minimum_branch), minimum_branch.shape)
    peak_q_index = int(np.argmax(negative_fraction))
    print(f"Saved {args.output}")
    print(f"Saved {args.output_csv}")
    print(f"Frames with imaginary modes: {unstable_frames}/{len(iteration)}")
    print(f"Sampled q-points ever imaginary: {unstable_qpoints}/{len(x_values)}")
    print(
        "Most negative mode: "
        f"{minimum_branch[minimum_index]:.6f} THz at iteration "
        f"{iteration[minimum_index[0]]}, q index {minimum_index[1]}"
    )
    print(
        "Highest occurrence: "
        f"{100.0 * negative_fraction[peak_q_index]:.2f}% at q index {peak_q_index}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
