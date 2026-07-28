#!/usr/bin/env python3
"""Animate imaginary-frequency count versus temperature by MD iteration."""

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
DEFAULT_DATA = CASE_DIR / "held_imaginary_count_vs_temperature.csv"
DEFAULT_OUTPUT = CASE_DIR / "held_imaginary_count_vs_temperature_by_step.gif"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--fps", type=int, default=12)
    parser.add_argument("--dpi", type=int, default=100)
    return parser.parse_args()


def padded_limits(values: np.ndarray, fraction: float = 0.05) -> tuple[float, float]:
    low = float(np.min(values))
    high = float(np.max(values))
    padding = fraction * (high - low)
    return low - padding, high + padding


def binned_statistics(
    temperature: np.ndarray, negative_count: np.ndarray, bins: int = 12
) -> tuple[np.ndarray, np.ndarray]:
    edges = np.linspace(float(np.min(temperature)), float(np.max(temperature)), bins + 1)
    indices = np.clip(np.digitize(temperature, edges) - 1, 0, bins - 1)
    centers = 0.5 * (edges[:-1] + edges[1:])
    means = np.full(bins, np.nan)
    for index in range(bins):
        values = negative_count[indices == index]
        if len(values):
            means[index] = float(np.mean(values))
    finite = np.isfinite(means)
    return centers[finite], means[finite]


def main() -> int:
    args = parse_args()
    table = np.atleast_1d(np.genfromtxt(args.data, delimiter=",", names=True))
    iteration = np.asarray(table["iteration"], dtype=int)
    temperature = np.asarray(table["temperature_K"], dtype=float)
    negative_count = np.asarray(table["negative_q_branch_count"], dtype=int)
    bin_temperature, bin_mean = binned_statistics(temperature, negative_count)

    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11})
    figure, axis = plt.subplots(figsize=(10.5, 7.0))
    figure.subplots_adjust(left=0.13, right=0.96, bottom=0.13, top=0.81)
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
        "BCC Fe Nose-Hoover AIMD  |  frequency < −0.01 THz  |  903 q-branch samples",
        ha="left",
        color="#5f6b7a",
        fontsize=10.5,
    )
    status = figure.text(
        0.13,
        0.855,
        "",
        ha="left",
        color="#0b2d4d",
        fontsize=11.5,
        fontweight="bold",
    )

    axis.scatter(
        temperature,
        negative_count,
        color="#94a3b8",
        s=25,
        alpha=0.14,
        linewidths=0,
        label="All iterations",
    )
    axis.plot(
        bin_temperature,
        bin_mean,
        color="#b42318",
        linewidth=1.8,
        marker="o",
        markersize=4,
        alpha=0.70,
        label="Binned mean",
    )
    history_line = axis.plot(
        [],
        [],
        color="#2878b5",
        linewidth=1.0,
        alpha=0.35,
        zorder=2,
    )[0]
    history_points = axis.scatter(
        [],
        [],
        color="#2878b5",
        s=28,
        alpha=0.55,
        linewidths=0,
        label="Previous iterations",
        zorder=3,
    )
    current_point = axis.scatter(
        [],
        [],
        color="#f47c20",
        s=150,
        edgecolors="white",
        linewidths=1.5,
        label="Current iteration",
        zorder=5,
    )

    axis.set_xlim(*padded_limits(temperature))
    axis.set_ylim(0.0, padded_limits(negative_count)[1])
    axis.set_xlabel("Temperature (K)")
    axis.set_ylabel("Absolute number of negative frequencies\n(out of 903 q-branch samples)")
    axis.grid(color="#d8dee7", linewidth=0.75, alpha=0.70)
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    axis.legend(loc="upper right", frameon=False)

    def update(frame_index: int):
        history = slice(0, frame_index + 1)
        offsets = np.column_stack([temperature[history], negative_count[history]])
        history_points.set_offsets(offsets)
        history_line.set_data(temperature[history], negative_count[history])
        current_point.set_offsets(
            np.array([[temperature[frame_index], negative_count[frame_index]]])
        )
        status.set_text(
            f"MD iteration {iteration[frame_index]:03d}/{iteration[-1]:03d}"
            f"   •   T = {temperature[frame_index]:.1f} K"
            f"   •   imaginary frequencies = {negative_count[frame_index]}/903"
        )
        return history_line, history_points, current_point, status

    animation = FuncAnimation(
        figure,
        update,
        frames=len(iteration),
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
                "title": "BCC Fe imaginary-frequency count versus temperature",
                "artist": "HELD",
            },
        ),
        dpi=args.dpi,
    )
    plt.close(figure)
    print(f"Saved {len(iteration)} frames to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
