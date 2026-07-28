#!/usr/bin/env python3
"""Parse the QE Nose-Hoover trajectory and run the complete BCC HELD workflow."""

from __future__ import annotations

import importlib.util
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-held-nose")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp")

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt


CASE_DIR = Path(__file__).resolve().parent
HELD_ROOT = CASE_DIR.parents[2]
if str(HELD_ROOT) not in sys.path:
    sys.path.insert(0, str(HELD_ROOT))

from HELD import fit_case, plot_heatmap, plot_mean_dispersion


QE_OUTPUT = CASE_DIR / "fe1.out"
PARSED_NPZ = CASE_DIR / "simulation_parsed.npz"
HELD_INPUT = CASE_DIR / "simulation_held.npz"
HELD_CSV = CASE_DIR / "held_mean_all_valid.csv"
STEP_CSV = CASE_DIR / "held_steps_with_thermodynamics.csv"
DISPERSION_DATA = CASE_DIR / "held_dispersion_all_valid_mean.dat"
DISPERSION_PLOT = CASE_DIR / "held_dispersion_all_valid_mean.png"
HEATMAP_CACHE = CASE_DIR / "held_heatmap_steps_all_valid.npz"
HEATMAP_PLOT = CASE_DIR / "held_heatmap_all_valid_mean.png"
OBSERVABLES_PLOT = CASE_DIR / "held_temperature_pressure_energy_by_step.png"
SUMMARY_JSON = CASE_DIR / "held_run_summary.json"
ANIMATION_GIF = CASE_DIR / "bcc_Fe_nose_5000K_HELD_0001_0399.gif"
BOHR_TO_ANG = 0.529177210903


def load_qe_parser():
    parser_path = CASE_DIR / "qe_output_parser.py"
    spec = importlib.util.spec_from_file_location("qe_data_compress", parser_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load QE parser from {parser_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def prepare_held_input() -> dict[str, np.ndarray]:
    parser = load_qe_parser()
    data = parser.parse_qe_aimd_output(QE_OUTPUT, require_forces=True)
    if data is None:
        raise ValueError(f"No MD frames found in {QE_OUTPUT}")
    parser.save_archive_npz(PARSED_NPZ, data)

    payload = {key: value for key, value in data.items() if isinstance(value, np.ndarray)}
    cell_ang = (
        np.asarray(data["initial_cell_alat"], dtype=np.float64)
        * float(data["alat_bohr"])
        * BOHR_TO_ANG
    )
    payload["input_cell_parameters"] = cell_ang.astype(np.float32)
    payload["input_cell_unit"] = np.asarray("angstrom", dtype="U16")
    payload["symbols"] = np.full(int(data["natoms"]), "Fe", dtype="U2")
    payload["species"] = np.full(int(data["natoms"]), "Fe", dtype="U2")
    np.savez_compressed(HELD_INPUT, **payload)
    return payload


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


def plot_observables(table: np.ndarray, energy_ry: np.ndarray) -> None:
    iteration = np.asarray(table["iteration"], dtype=float)
    series = (
        ("temperature_K", "Temperature (K)", "#c94f2d"),
        ("pressure_GPa", "Pressure (GPa)", "#2878b5"),
        (energy_ry, "Total energy (Ry)", "#2a9d68"),
    )
    fig, axes = plt.subplots(3, 1, figsize=(10.5, 8.5), sharex=True, constrained_layout=True)
    for axis, (field_or_values, label, color) in zip(axes, series):
        values = (
            np.asarray(table[field_or_values], dtype=float)
            if isinstance(field_or_values, str)
            else np.asarray(field_or_values, dtype=float)
        )
        axis.plot(iteration, values, color=color, linewidth=1.4)
        axis.axhline(np.nanmean(values), color=color, linestyle="--", linewidth=0.9, alpha=0.65)
        axis.set_ylabel(label)
        axis.grid(alpha=0.22)
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)
    axes[-1].set_xlabel("MD iteration")
    fig.suptitle("BCC Fe Nose-Hoover AIMD observables")
    fig.savefig(OBSERVABLES_PLOT, dpi=220)
    plt.close(fig)


def main() -> int:
    payload = prepare_held_input()
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

    table = np.atleast_1d(np.genfromtxt(STEP_CSV, delimiter=",", names=True))
    all_iterations = np.asarray(payload["iteration"], dtype=int)
    valid = np.asarray(payload["frame_valid"], dtype=bool)
    energy_ry = np.asarray(payload["energy_ry"], dtype=float)[valid]
    plot_observables(table, energy_ry)
    cell_ang = np.asarray(payload["input_cell_parameters"], dtype=float)
    frequencies = np.asarray(dispersion["held_thz"], dtype=float)
    step_frequencies = np.asarray(heatmap["step_frequencies_thz"], dtype=float)
    summary = {
        "phase": "bcc",
        "material": "Fe",
        "thermostat": "Nose-Hoover",
        "target_temperature_K": 5000,
        "source_qe_output": QE_OUTPUT.name,
        "natoms": int(payload["positions"].shape[1]),
        "supercell": [4, 4, 4],
        "cell_length_ang": float(cell_ang[0, 0]),
        "conventional_lattice_parameter_ang": float(cell_ang[0, 0] / 4.0),
        "printed_md_frames": int(len(all_iterations)),
        "selected_complete_frames": int(info["n_frames"]),
        "first_iteration": int(result.step_ids[0]),
        "last_iteration": int(result.step_ids[-1]),
        "incomplete_iterations": [int(value) for value in all_iterations[~valid]],
        "aggregate": "arithmetic mean",
        "num_shells": int(info["num_shells"]),
        "shell_distances_ang": [float(value) for value in info["selected_shell_distances"]],
        "temperature_K": finite_statistics(table["temperature_K"]),
        "pressure_GPa": finite_statistics(table["pressure_GPa"]),
        "energy_Ry": finite_statistics(energy_ry),
        "mean_dispersion_min_THz": float(np.min(frequencies)),
        "mean_dispersion_max_THz": float(np.max(frequencies)),
        "negative_mean_path_points": int(np.count_nonzero(frequencies < -1.0e-6)),
        "per_step_dispersion_min_THz": float(np.min(step_frequencies)),
        "per_step_dispersion_max_THz": float(np.max(step_frequencies)),
        "heatmap_shape": [int(value) for value in step_frequencies.shape],
        "outputs": {
            "parsed_npz": PARSED_NPZ.name,
            "held_input_npz": HELD_INPUT.name,
            "coefficient_csv": HELD_CSV.name,
            "step_csv": STEP_CSV.name,
            "dispersion_data": DISPERSION_DATA.name,
            "dispersion_plot": DISPERSION_PLOT.name,
            "heatmap_cache": HEATMAP_CACHE.name,
            "heatmap_plot": HEATMAP_PLOT.name,
            "observables_plot": OBSERVABLES_PLOT.name,
            "animation_gif": ANIMATION_GIF.name,
        },
    }
    SUMMARY_JSON.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
