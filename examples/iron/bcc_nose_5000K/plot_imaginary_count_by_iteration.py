#!/usr/bin/env python3
"""Integrate imaginary HELD frequencies over the symmetry path per MD iteration."""

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


CASE_DIR = Path(__file__).resolve().parent
DEFAULT_CACHE = CASE_DIR / "held_heatmap_steps_all_valid.npz"
DEFAULT_STEPS = CASE_DIR / "held_steps_with_thermodynamics.csv"
DEFAULT_PLOT = CASE_DIR / "held_imaginary_count_by_iteration.png"
DEFAULT_CSV = CASE_DIR / "held_imaginary_count_by_iteration.csv"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--steps-csv", type=Path, default=DEFAULT_STEPS)
    parser.add_argument("--output", type=Path, default=DEFAULT_PLOT)
    parser.add_argument("--output-csv", type=Path, default=DEFAULT_CSV)
    parser.add_argument("--threshold-thz", type=float, default=0.01)
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
    time_ps = np.asarray(table["time_ps"], dtype=float)
    if len(iteration) != len(frequencies):
        raise ValueError("Step table and frequency-cache frame counts do not match")

    imaginary = frequencies < -args.threshold_thz
    absolute_count = np.count_nonzero(imaginary, axis=(1, 2))
    branch_counts = np.count_nonzero(imaginary, axis=1)
    qpoint_count = np.count_nonzero(np.any(imaginary, axis=2), axis=1)

    path_length = float(x_values[-1] - x_values[0])
    indicator_integral = np.trapz(imaginary.astype(float), x=x_values, axis=1).sum(axis=1)
    integrated_path_fraction = indicator_integral / (frequencies.shape[2] * path_length)
    imaginary_magnitude = np.where(imaginary, -frequencies, 0.0)
    absolute_frequency_integral = np.trapz(
        imaginary_magnitude, x=x_values, axis=1
    ).sum(axis=1)

    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    np.savetxt(
        args.output_csv,
        np.column_stack(
            [
                iteration,
                time_ps,
                absolute_count,
                qpoint_count,
                branch_counts,
                indicator_integral,
                integrated_path_fraction,
                absolute_frequency_integral,
            ]
        ),
        delimiter=",",
        header=(
            "iteration,time_ps,negative_q_branch_count,negative_qpoint_count,"
            "branch_1_count,branch_2_count,branch_3_count,"
            "path_integral_negative_indicator,"
            "path_integrated_negative_fraction,"
            "path_integral_absolute_imaginary_frequency_THz"
        ),
        comments="",
        fmt=["%d", "%.10f", "%d", "%d", "%d", "%d", "%d", "%.10f", "%.10f", "%.10f"],
    )

    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10.5})
    figure, axes = plt.subplots(
        3, 1, figsize=(12.0, 9.5), sharex=True,
        gridspec_kw={"height_ratios": (1.15, 1.0, 1.0)},
        constrained_layout=True,
    )
    figure.suptitle(
        "BCC Fe: Imaginary HELD Modes Integrated over the Symmetry Path",
        x=0.08, ha="left", color="#0b2d4d", fontsize=18, fontweight="bold",
    )
    figure.text(
        0.08, 0.94,
        (
            f"Threshold: frequency < −{args.threshold_thz:.2f} THz  |  "
            f"{frequencies.shape[1]} q-points × {frequencies.shape[2]} branches "
            f"= {frequencies.shape[1] * frequencies.shape[2]} samples per iteration"
        ),
        ha="left", color="#5f6b7a", fontsize=10.5,
    )

    axes[0].fill_between(iteration, absolute_count, color="#ef4444", alpha=0.22)
    axes[0].plot(iteration, absolute_count, color="#b42318", linewidth=1.5)
    axes[0].set_ylabel("Absolute number of\nnegative frequencies")
    axes[0].set_ylim(bottom=0)
    axes[0].text(
        0.99, 0.92,
        f"maximum = {int(np.max(absolute_count))}/903",
        transform=axes[0].transAxes, ha="right", va="top", color="#7f1d1d",
    )

    axes[1].fill_between(
        iteration, 100.0 * integrated_path_fraction,
        color="#f47c20", alpha=0.22,
    )
    axes[1].plot(
        iteration, 100.0 * integrated_path_fraction,
        color="#d95f02", linewidth=1.4,
    )
    axes[1].set_ylabel("Path-weighted imaginary\nfraction (%)")
    axes[1].set_ylim(bottom=0)

    axes[2].fill_between(
        iteration, absolute_frequency_integral,
        color="#7b4ab5", alpha=0.20,
    )
    axes[2].plot(
        iteration, absolute_frequency_integral,
        color="#6a3d9a", linewidth=1.4,
    )
    axes[2].set_ylabel(r"$\int \sum_b |\nu_b^-|\,dq$")
    axes[2].set_xlabel("MD iteration")
    axes[2].set_ylim(bottom=0)

    for axis in axes:
        axis.grid(color="#d8dee7", linewidth=0.7, alpha=0.70)
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)
        axis.set_xlim(float(iteration[0]), float(iteration[-1]))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.output, dpi=220)
    plt.close(figure)

    maximum_index = int(np.argmax(absolute_count))
    print(f"Saved {args.output}")
    print(f"Saved {args.output_csv}")
    print(
        f"Maximum absolute count: {absolute_count[maximum_index]}/"
        f"{frequencies.shape[1] * frequencies.shape[2]} at iteration "
        f"{iteration[maximum_index]}"
    )
    print(
        f"Iterations with at least one imaginary frequency: "
        f"{np.count_nonzero(absolute_count)}/{len(iteration)}"
    )
    print(f"Mean absolute count: {np.mean(absolute_count):.6f}")
    print(
        f"Maximum path-weighted fraction: "
        f"{100.0 * np.max(integrated_path_fraction):.6f}%"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
