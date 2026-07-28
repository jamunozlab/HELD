#!/usr/bin/env python3
"""Plot the absolute imaginary-frequency count against temperature."""

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
DEFAULT_COUNTS = CASE_DIR / "held_imaginary_count_by_iteration.csv"
DEFAULT_STEPS = CASE_DIR / "held_steps_with_thermodynamics.csv"
DEFAULT_PLOT = CASE_DIR / "held_imaginary_count_vs_temperature.png"
DEFAULT_CSV = CASE_DIR / "held_imaginary_count_vs_temperature.csv"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--counts-csv", type=Path, default=DEFAULT_COUNTS)
    parser.add_argument("--steps-csv", type=Path, default=DEFAULT_STEPS)
    parser.add_argument("--output", type=Path, default=DEFAULT_PLOT)
    parser.add_argument("--output-csv", type=Path, default=DEFAULT_CSV)
    parser.add_argument("--bins", type=int, default=12)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    counts = np.atleast_1d(np.genfromtxt(args.counts_csv, delimiter=",", names=True))
    steps = np.atleast_1d(np.genfromtxt(args.steps_csv, delimiter=",", names=True))
    count_iteration = np.asarray(counts["iteration"], dtype=int)
    step_iteration = np.asarray(steps["iteration"], dtype=int)
    if not np.array_equal(count_iteration, step_iteration):
        raise ValueError("Iteration columns in the count and thermodynamic tables do not match")

    temperature = np.asarray(steps["temperature_K"], dtype=float)
    negative_count = np.asarray(counts["negative_q_branch_count"], dtype=int)
    finite = np.isfinite(temperature)
    iteration = count_iteration[finite]
    temperature = temperature[finite]
    negative_count = negative_count[finite]
    correlation = float(np.corrcoef(temperature, negative_count)[0, 1])

    bin_edges = np.linspace(float(np.min(temperature)), float(np.max(temperature)), args.bins + 1)
    bin_index = np.clip(np.digitize(temperature, bin_edges) - 1, 0, args.bins - 1)
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    bin_mean = np.full(args.bins, np.nan)
    bin_q25 = np.full(args.bins, np.nan)
    bin_q75 = np.full(args.bins, np.nan)
    for index in range(args.bins):
        values = negative_count[bin_index == index]
        if len(values):
            bin_mean[index] = float(np.mean(values))
            bin_q25[index], bin_q75[index] = np.percentile(values, [25.0, 75.0])

    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    np.savetxt(
        args.output_csv,
        np.column_stack([iteration, temperature, negative_count]),
        delimiter=",",
        header="iteration,temperature_K,negative_q_branch_count",
        comments="",
        fmt=["%d", "%.8f", "%d"],
    )

    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11})
    figure, axis = plt.subplots(figsize=(10.5, 7.0))
    figure.subplots_adjust(left=0.13, right=0.86, bottom=0.13, top=0.82)
    scatter = axis.scatter(
        temperature,
        negative_count,
        c=iteration,
        cmap="viridis",
        s=35,
        alpha=0.78,
        linewidths=0.25,
        edgecolors="white",
        label="MD iterations",
    )
    valid_bins = np.isfinite(bin_mean)
    axis.fill_between(
        bin_centers[valid_bins],
        bin_q25[valid_bins],
        bin_q75[valid_bins],
        color="#d9483b",
        alpha=0.18,
        label="Binned interquartile range",
    )
    axis.plot(
        bin_centers[valid_bins],
        bin_mean[valid_bins],
        color="#b42318",
        linewidth=2.2,
        marker="o",
        markersize=5,
        label="Binned mean",
    )
    colorbar = figure.colorbar(scatter, ax=axis, pad=0.02)
    colorbar.set_label("MD iteration")

    figure.suptitle(
        "Instantaneous Imaginary-Frequency Count vs Temperature",
        x=0.13,
        y=0.965,
        ha="left",
        color="#0b2d4d",
        fontsize=17,
        fontweight="bold",
    )
    figure.text(
        0.13,
        0.905,
        (
            "BCC Fe Nose-Hoover AIMD  |  frequency < −0.01 THz  |  "
            f"Pearson r = {correlation:.3f}"
        ),
        ha="left",
        color="#5f6b7a",
        fontsize=10.5,
    )
    axis.set_xlabel("Temperature (K)")
    axis.set_ylabel("Absolute number of negative frequencies\n(out of 903 q-branch samples)")
    axis.set_ylim(bottom=0)
    axis.grid(color="#d8dee7", linewidth=0.75, alpha=0.70)
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    axis.legend(loc="upper right", frameon=False)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.output, dpi=220)
    plt.close(figure)
    print(f"Saved {args.output}")
    print(f"Saved {args.output_csv}")
    print(f"Pearson correlation: {correlation:.8f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
