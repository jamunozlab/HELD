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

from HELD import fit_case, plot_mean_dispersion
from held.model import build_model_from_npz
from held.phases import build_q_path


SOURCE_NPZ = WORKSPACE_ROOT / "dataset" / "bcc" / "magnetic-non_coll" / "simulation.npz"
IDEAL_NPZ = WORKSPACE_ROOT / "dataset" / "bcc" / "magnetic-non_coll" / "simulation-ideal.npz"
OUTPUT_DIR = Path(__file__).resolve().parent
HELD_INPUT = OUTPUT_DIR / "simulation_held_ideal_reference.npz"
HELD_CSV = OUTPUT_DIR / "held_mean_two_frames.csv"
STEP_CSV = OUTPUT_DIR / "held_steps_with_thermodynamics.csv"
MEAN_DATA = OUTPUT_DIR / "held_dispersion_two_frame_mean.dat"
MEAN_PLOT = OUTPUT_DIR / "held_dispersion_two_frame_mean.png"
SNAPSHOT_DATA = OUTPUT_DIR / "held_dispersion_two_snapshots.npz"
SNAPSHOT_PLOT = OUTPUT_DIR / "held_dispersion_two_snapshots_and_mean.png"
SUMMARY_JSON = OUTPUT_DIR / "held_run_summary.json"


def prepare_held_input() -> None:
    with np.load(SOURCE_NPZ, allow_pickle=False) as source, np.load(
        IDEAL_NPZ, allow_pickle=False
    ) as ideal:
        payload = {name: source[name] for name in source.files}
        payload["initial_positions_alat"] = ideal["initial_positions_alat"]
        payload["initial_cell_alat"] = ideal["initial_cell_alat"]
        payload["symbols"] = np.full(source["positions"].shape[1], "Fe", dtype="U2")
        payload["species"] = np.full(source["positions"].shape[1], "Fe", dtype="U2")
        np.savez_compressed(HELD_INPUT, **payload)


def plot_snapshot_dispersions(
    x_values: np.ndarray,
    tick_labels: list[str],
    tick_positions: np.ndarray,
    snapshot_frequencies: np.ndarray,
    mean_frequencies: np.ndarray,
    step_ids: list[int],
) -> None:
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 12})
    fig, axis = plt.subplots(figsize=(10.5, 6.7), constrained_layout=True)
    snapshot_colors = ("#2a78b8", "#d46a1f")
    snapshot_labels = ("Initial configuration", f"MD iteration {step_ids[1]}")

    for tick_position in tick_positions:
        axis.axvline(tick_position, color="#cbd5e1", linewidth=0.8, zorder=0)
    axis.axhline(0.0, color="#1f2937", linewidth=0.9)

    for snapshot_index, (color, label) in enumerate(
        zip(snapshot_colors, snapshot_labels)
    ):
        for branch_index in range(snapshot_frequencies.shape[2]):
            axis.plot(
                x_values,
                snapshot_frequencies[snapshot_index, :, branch_index],
                color=color,
                linewidth=1.25,
                alpha=0.78,
                label=label if branch_index == 0 else None,
            )

    for branch_index in range(mean_frequencies.shape[1]):
        axis.plot(
            x_values,
            mean_frequencies[:, branch_index],
            color="#111827",
            linewidth=2.15,
            label="Two-frame mean" if branch_index == 0 else None,
        )

    all_frequencies = np.concatenate(
        [snapshot_frequencies.reshape(-1), mean_frequencies.reshape(-1)]
    )
    finite = all_frequencies[np.isfinite(all_frequencies)]
    minimum = float(np.min(finite))
    maximum = float(np.max(finite))
    padding = 0.06 * (maximum - minimum)
    axis.set_ylim(min(0.0, minimum - padding), maximum + padding)
    axis.set_xlim(float(x_values[0]), float(x_values[-1]))
    axis.set_xticks(tick_positions)
    axis.set_xticklabels(tick_labels)
    axis.set_ylabel("Frequency (THz)")
    axis.set_title(
        "BCC Fe Noncollinear HELD Phonons at 4000 K",
        fontsize=15,
        fontweight="bold",
    )
    axis.text(
        0.01,
        0.98,
        "Diagnostic fit from two complete force snapshots",
        transform=axis.transAxes,
        ha="left",
        va="top",
        color="#5b6472",
        fontsize=10.5,
    )
    axis.grid(axis="y", alpha=0.22)
    axis.legend(frameon=False, loc="upper right")
    fig.savefig(SNAPSHOT_PLOT, dpi=220)
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
    mean_dispersion = plot_mean_dispersion(
        phase="bcc",
        npz_path=HELD_INPUT,
        held_csv=HELD_CSV,
        output_data=MEAN_DATA,
        output_plot=MEAN_PLOT,
        points_per_segment=60,
    )

    model, _dataset, _metadata = build_model_from_npz(
        phase="bcc", npz_path=HELD_INPUT
    )
    q_path, x_values, tick_labels, tick_positions = build_q_path(
        phase="bcc", primitive_cell=model.uc_cell, points_per_segment=60
    )
    snapshot_frequencies = np.stack(
        [
            model.dispersion_thz_from_reduced_path(coefficients, q_path)
            for coefficients in result.step_values
        ]
    )
    mean_frequencies = np.asarray(mean_dispersion["held_thz"], dtype=float)
    np.savez_compressed(
        SNAPSHOT_DATA,
        x_values=x_values,
        tick_positions=tick_positions,
        tick_labels=np.asarray(tick_labels, dtype="U8"),
        step_ids=np.asarray(result.step_ids, dtype=np.int32),
        snapshot_frequencies_thz=snapshot_frequencies,
        mean_frequencies_thz=mean_frequencies,
    )
    plot_snapshot_dispersions(
        x_values=x_values,
        tick_labels=tick_labels,
        tick_positions=tick_positions,
        snapshot_frequencies=snapshot_frequencies,
        mean_frequencies=mean_frequencies,
        step_ids=result.step_ids,
    )

    table = np.genfromtxt(STEP_CSV, delimiter=",", names=True)
    branch_spread = np.std(snapshot_frequencies, axis=0)
    summary = {
        "phase": info["phase"],
        "magnetic_state": "noncollinear",
        "temperature_label_K": 4000,
        "source_npz": str(SOURCE_NPZ),
        "ideal_reference_npz": str(IDEAL_NPZ),
        "held_input_npz": str(HELD_INPUT),
        "aggregate": "arithmetic mean",
        "selected_frames": int(info["n_frames"]),
        "step_ids": [int(value) for value in result.step_ids],
        "natoms": 128,
        "num_shells": int(info["num_shells"]),
        "num_coefficients": int(len(result.labels)),
        "shell_distances_ang": [
            float(value) for value in info["selected_shell_distances"]
        ],
        "temperature_K": [float(value) for value in np.atleast_1d(table["temperature_K"])],
        "pressure_GPa": [float(value) for value in np.atleast_1d(table["pressure_GPa"])],
        "total_magnetization_Bohr_per_cell": [
            float(value) for value in np.atleast_1d(table["mag_total_Bohr"])
        ],
        "absolute_magnetization_Bohr_per_cell": [
            float(value) for value in np.atleast_1d(table["abs_mag_total_Bohr"])
        ],
        "mean_dispersion_min_THz": float(np.min(mean_frequencies)),
        "mean_dispersion_max_THz": float(np.max(mean_frequencies)),
        "snapshot_dispersion_min_THz": [
            float(np.min(values)) for values in snapshot_frequencies
        ],
        "snapshot_dispersion_max_THz": [
            float(np.max(values)) for values in snapshot_frequencies
        ],
        "maximum_snapshot_standard_deviation_THz": float(np.max(branch_spread)),
        "negative_mean_path_points": int(np.count_nonzero(mean_frequencies < -1.0e-6)),
        "limitations": [
            "Only two complete force snapshots are available.",
            "The fit is algebraically overdetermined but not statistically converged.",
            "Snapshot spread must be retained when interpreting the two-frame mean.",
            "Additional decorrelated AIMD frames are required for quantitative phonons.",
        ],
        "outputs": {
            "coefficient_csv": str(HELD_CSV),
            "step_csv": str(STEP_CSV),
            "mean_dispersion_data": str(MEAN_DATA),
            "mean_dispersion_plot": str(MEAN_PLOT),
            "snapshot_dispersion_data": str(SNAPSHOT_DATA),
            "snapshot_comparison_plot": str(SNAPSHOT_PLOT),
        },
    }
    SUMMARY_JSON.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
