#!/usr/bin/env python3

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-held")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp")

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt


PROJECT_ROOT = Path(__file__).resolve().parents[3]
WORKSPACE_ROOT = PROJECT_ROOT.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from HELD import fit_case, plot_heatmap, plot_mean_dispersion


SOURCE_NPZ = WORKSPACE_ROOT / "dataset" / "bcc" / "magnetic-collinear" / "simulation.npz"
IDEAL_NPZ = WORKSPACE_ROOT / "dataset" / "bcc" / "magnetic-collinear" / "simulation-ideal.npz"
OUTPUT_DIR = Path(__file__).resolve().parent
HELD_INPUT = OUTPUT_DIR / "simulation_held_ideal_reference.npz"
HELD_CSV = OUTPUT_DIR / "held_mean_all_valid.csv"
STEP_CSV = OUTPUT_DIR / "held_steps_with_thermodynamics.csv"
DISPERSION_DATA = OUTPUT_DIR / "held_dispersion_all_valid_mean.dat"
DISPERSION_PLOT = OUTPUT_DIR / "held_dispersion_all_valid_mean.png"
HEATMAP_CACHE = OUTPUT_DIR / "held_heatmap_steps_all_valid.npz"
HEATMAP_PLOT = OUTPUT_DIR / "held_heatmap_all_valid_mean.png"
OBSERVABLES_PLOT = OUTPUT_DIR / "held_temperature_pressure_magnetization_by_step.png"
SUMMARY_JSON = OUTPUT_DIR / "held_run_summary.json"


def prepare_held_input() -> None:
    with np.load(SOURCE_NPZ, allow_pickle=False) as source, np.load(IDEAL_NPZ, allow_pickle=False) as ideal:
        payload = {name: source[name] for name in source.files}
        payload["initial_positions_alat"] = ideal["initial_positions_alat"]
        payload["initial_cell_alat"] = ideal["initial_cell_alat"]
        payload["symbols"] = np.full(source["positions"].shape[1], "Fe", dtype="U2")
        payload["species"] = np.full(source["positions"].shape[1], "Fe", dtype="U2")
        np.savez_compressed(HELD_INPUT, **payload)


def finite_statistics(values: np.ndarray) -> dict[str, float | int]:
    finite = np.asarray(values, dtype=float)
    finite = finite[np.isfinite(finite)]
    return {
        "count": int(len(finite)),
        "mean": float(np.mean(finite)),
        "std": float(np.std(finite)),
        "min": float(np.min(finite)),
        "max": float(np.max(finite)),
    }


def plot_observables() -> None:
    table = np.genfromtxt(STEP_CSV, delimiter=",", names=True)
    iteration = table["iteration"]
    fig, axes = plt.subplots(3, 1, figsize=(10.5, 9.0), sharex=True, constrained_layout=True)

    axes[0].plot(iteration, table["temperature_K"], color="#d95f02", linewidth=1.5)
    axes[0].axhline(np.nanmean(table["temperature_K"]), color="#7f2704", linestyle="--", linewidth=1.0)
    axes[0].set_ylabel("Temperature (K)")

    axes[1].plot(iteration, table["pressure_GPa"], color="#1f78b4", linewidth=1.5)
    axes[1].axhline(np.nanmean(table["pressure_GPa"]), color="#08306b", linestyle="--", linewidth=1.0)
    axes[1].set_ylabel("Pressure (GPa)")

    axes[2].plot(
        iteration,
        table["mag_total_Bohr"],
        color="#6a3d9a",
        linewidth=1.5,
        label="Total magnetization",
    )
    axes[2].plot(
        iteration,
        table["abs_mag_total_Bohr"],
        color="#33a02c",
        linewidth=1.5,
        label="Absolute magnetization",
    )
    axes[2].set_ylabel("Magnetization\n(Bohr magnetons/cell)")
    axes[2].set_xlabel("MD iteration")
    axes[2].legend(frameon=False, ncol=2)

    for axis in axes:
        axis.grid(alpha=0.22)
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)

    fig.savefig(OBSERVABLES_PLOT, dpi=220)
    plt.close(fig)


def main() -> int:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    prepare_held_input()

    result, info = fit_case(
        phase="bcc",
        npz_path=HELD_INPUT,
        output_csv=HELD_CSV,
        output_steps_csv=STEP_CSV,
        aggregate="mean",
        skip=0,
        every=1,
        verbose=True,
    )
    dispersion = plot_mean_dispersion(
        phase="bcc",
        npz_path=HELD_INPUT,
        held_csv=HELD_CSV,
        output_data=DISPERSION_DATA,
        output_plot=DISPERSION_PLOT,
        points_per_segment=60,
    )
    heatmap = plot_heatmap(
        phase="bcc",
        npz_path=HELD_INPUT,
        held_csv=HELD_CSV,
        cache_npz=HEATMAP_CACHE,
        output_plot=HEATMAP_PLOT,
        points_per_segment=60,
        y_bins=600,
        force_recompute=True,
        verbose=True,
    )
    plot_observables()

    table = np.genfromtxt(STEP_CSV, delimiter=",", names=True)
    summary = {
        "phase": info["phase"],
        "symbol": info["symbol"],
        "source_npz": str(SOURCE_NPZ),
        "ideal_reference_npz": str(IDEAL_NPZ),
        "held_input_npz": str(HELD_INPUT),
        "aggregate": "mean",
        "selected_frames": int(info["n_frames"]),
        "first_iteration": int(result.step_ids[0]),
        "last_iteration": int(result.step_ids[-1]),
        "num_shells": int(info["num_shells"]),
        "shell_distances_ang": [float(value) for value in info["selected_shell_distances"]],
        "temperature_K": finite_statistics(table["temperature_K"]),
        "pressure_GPa": finite_statistics(table["pressure_GPa"]),
        "total_magnetization_Bohr_per_cell": finite_statistics(table["mag_total_Bohr"]),
        "absolute_magnetization_Bohr_per_cell": finite_statistics(table["abs_mag_total_Bohr"]),
        "mean_dispersion_min_THz": float(np.min(dispersion["held_thz"])),
        "mean_dispersion_max_THz": float(np.max(dispersion["held_thz"])),
        "heatmap_shape": [int(value) for value in heatmap["step_frequencies_thz"].shape],
        "outputs": {
            "coefficient_csv": str(HELD_CSV),
            "step_csv": str(STEP_CSV),
            "dispersion_data": str(DISPERSION_DATA),
            "dispersion_plot": str(DISPERSION_PLOT),
            "heatmap_cache": str(HEATMAP_CACHE),
            "heatmap_plot": str(HEATMAP_PLOT),
            "observables_plot": str(OBSERVABLES_PLOT),
        },
    }
    SUMMARY_JSON.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
