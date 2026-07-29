#!/usr/bin/env python3

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-held")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp")

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[3]
WORKSPACE_ROOT = PROJECT_ROOT.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from HELD import fit_case, plot_heatmap, plot_mean_dispersion


SOURCE_NPZ = WORKSPACE_ROOT / "dataset" / "bcc" / "magnetic-non_coll" / "simulation.npz"
OUTPUT_DIR = Path(__file__).resolve().parent
HELD_INPUT = OUTPUT_DIR / "simulation_held_ideal_reference.npz"
HELD_CSV = OUTPUT_DIR / "held_mean_all_complete.csv"
STEP_CSV = OUTPUT_DIR / "held_steps_with_thermodynamics.csv"
DISPERSION_DATA = OUTPUT_DIR / "held_dispersion_all_complete_mean.dat"
DISPERSION_PLOT = OUTPUT_DIR / "held_dispersion_all_complete_mean.png"
HEATMAP_CACHE = OUTPUT_DIR / "held_heatmap_steps_all_complete.npz"
HEATMAP_PLOT = OUTPUT_DIR / "held_heatmap_all_complete.png"
SUMMARY_JSON = OUTPUT_DIR / "held_run_summary.json"


def ideal_bcc_fractional(natoms: int) -> np.ndarray:
    repetitions = round((natoms / 2.0) ** (1.0 / 3.0))
    if 2 * repetitions**3 != natoms:
        raise ValueError(f"Cannot construct a conventional BCC supercell for {natoms} atoms")
    corners = []
    centers = []
    for z_index in range(repetitions):
        for y_index in range(repetitions):
            for x_index in range(repetitions):
                corners.append([x_index / repetitions, y_index / repetitions, z_index / repetitions])
                centers.append(
                    [
                        (x_index + 0.5) / repetitions,
                        (y_index + 0.5) / repetitions,
                        (z_index + 0.5) / repetitions,
                    ]
                )
    return np.asarray(corners + centers, dtype=np.float32)


def prepare_held_input() -> None:
    with np.load(SOURCE_NPZ, allow_pickle=False) as source:
        payload = {name: source[name] for name in source.files}
        natoms = int(source["positions"].shape[1])
        ideal_fractional = ideal_bcc_fractional(natoms)
        cell_ang = np.eye(3, dtype=np.float32) * 7.65
        payload["initial_positions_alat"] = ideal_fractional
        payload["initial_cell_alat"] = np.eye(3, dtype=np.float32)
        payload["input_cell_parameters"] = cell_ang
        payload["input_cell_unit"] = np.asarray("angstrom", dtype="U16")
        payload["symbols"] = np.full(natoms, "Fe", dtype="U2")
        payload["species"] = np.full(natoms, "Fe", dtype="U2")
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

    table = np.atleast_1d(np.genfromtxt(STEP_CSV, delimiter=",", names=True))
    with np.load(SOURCE_NPZ, allow_pickle=False) as source:
        valid = np.asarray(source["frame_valid"], dtype=bool)
        magnetization_vector = np.asarray(source["mag_total_vector_Bohr"], dtype=float)[valid]
        magnetization_norm = np.linalg.norm(magnetization_vector, axis=1)
        times = np.asarray(source["time_ps"], dtype=float)[valid]

    frequencies = np.asarray(dispersion["held_thz"], dtype=float)
    summary = {
        "phase": "bcc",
        "magnetic_state": "noncollinear",
        "temperature_label_K": 4000,
        "source_npz": str(SOURCE_NPZ),
        "held_input_npz": str(HELD_INPUT),
        "natoms": 54,
        "supercell": [3, 3, 3],
        "selected_frames": int(info["n_frames"]),
        "first_iteration": int(result.step_ids[0]),
        "last_iteration": int(result.step_ids[-1]),
        "trajectory_span_ps": float(times[-1] - times[0]),
        "aggregate": "arithmetic mean",
        "num_shells": int(info["num_shells"]),
        "shell_distances_ang": [float(value) for value in info["selected_shell_distances"]],
        "temperature_K": finite_statistics(table["temperature_K"]),
        "pressure_GPa": finite_statistics(table["pressure_GPa"]),
        "global_magnetization_magnitude_Bohr_per_cell": finite_statistics(magnetization_norm),
        "absolute_magnetization_Bohr_per_cell": finite_statistics(table["abs_mag_total_Bohr"]),
        "mean_dispersion_min_THz": float(np.min(frequencies)),
        "mean_dispersion_max_THz": float(np.max(frequencies)),
        "negative_mean_path_points": int(np.count_nonzero(frequencies < -1.0e-6)),
        "heatmap_shape": [int(value) for value in heatmap["step_frequencies_thz"].shape],
        "limitations": [
            (
                f"The {info['n_frames']} complete frames are consecutive and span only "
                f"{times[-1] - times[0]:.3f} ps, so they are strongly correlated."
            ),
            "The 3x3x3 box is smaller than twice the fifth-shell force-constant cutoff used by the default BCC HELD model.",
            "This result is an early trajectory diagnostic, not a statistically converged finite-temperature phonon spectrum.",
            "Quantitative interpretation requires a longer equilibrated trajectory and a supercell-size/cutoff convergence test.",
        ],
        "outputs": {
            "coefficient_csv": str(HELD_CSV),
            "step_csv": str(STEP_CSV),
            "dispersion_data": str(DISPERSION_DATA),
            "dispersion_plot": str(DISPERSION_PLOT),
            "heatmap_cache": str(HEATMAP_CACHE),
            "heatmap_plot": str(HEATMAP_PLOT),
        },
    }
    SUMMARY_JSON.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
