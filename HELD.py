#!/usr/bin/env python3
"""Unified HELD API and CLI for BCC, FCC, and HCP NPZ trajectories."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp")
import numpy as np
from scipy.ndimage import gaussian_filter

from held.io import HeldRunResult, read_fc_csv, write_fc_csv
from held.model import build_model_from_npz
from held.phases import PHASES, build_q_path


def load_plotting():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap, PowerNorm

    return plt, LinearSegmentedColormap, PowerNorm


def fit_case(
    phase: str,
    npz_path: str | Path,
    output_csv: str | Path | None = None,
    aggregate: str = "mean",
    skip: int = 0,
    every: int = 1,
    max_frames: int = 0,
    num_shells: int | None = None,
    cutoff_ang: float | None = None,
    mass_amu: float | None = None,
    verbose: bool = False,
) -> tuple[HeldRunResult, dict[str, object]]:
    model, dataset, metadata = build_model_from_npz(
        phase=phase,
        npz_path=Path(npz_path),
        num_shells=num_shells,
        cutoff_ang=cutoff_ang,
        mass_amu=mass_amu,
    )
    positions_frac, forces_ev_ang, step_ids = dataset.select_frames(skip=skip, every=every, max_frames=max_frames)
    result = model.fit_series(positions_frac, forces_ev_ang, step_ids, aggregate=aggregate, verbose=verbose)
    if output_csv is not None:
        write_fc_csv(Path(output_csv), result)
    info = {
        "phase": metadata.phase,
        "symbol": metadata.symbol,
        "mass_amu": metadata.mass_amu,
        "num_shells": metadata.num_shells,
        "selected_shell_distances": metadata.selected_shell_distances,
        "n_frames": len(result.step_ids),
        "npz_path": str(npz_path),
    }
    return result, info


def _band_header(prefix: str, n_bands: int) -> list[str]:
    return [f"{prefix}_b{index + 1}_THz" for index in range(n_bands)]


def write_dispersion_data(path: Path, x_values: np.ndarray, held_thz: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    header = "x_path " + " ".join(_band_header("held", held_thz.shape[1]))
    np.savetxt(path, np.column_stack([x_values, held_thz]), fmt="%.10f", header=header)


def plot_mean_dispersion(
    phase: str,
    npz_path: str | Path,
    held_csv: str | Path,
    output_data: str | Path | None = None,
    output_plot: str | Path | None = None,
    num_shells: int | None = None,
    cutoff_ang: float | None = None,
    mass_amu: float | None = None,
    path_labels: list[str] | None = None,
    points_per_segment: int = 90,
) -> dict[str, object]:
    model, _dataset, metadata = build_model_from_npz(
        phase=phase,
        npz_path=Path(npz_path),
        num_shells=num_shells,
        cutoff_ang=cutoff_ang,
        mass_amu=mass_amu,
    )
    _labels, mean_values, _step_values = read_fc_csv(Path(held_csv))
    q_path, x_values, tick_labels, tick_positions = build_q_path(
        phase=phase,
        primitive_cell=model.uc_cell,
        path_labels=path_labels,
        points_per_segment=points_per_segment,
    )
    held_thz = model.dispersion_thz_from_reduced_path(mean_values, q_path)

    if output_data is not None:
        write_dispersion_data(Path(output_data), x_values, held_thz)

    if output_plot is not None:
        plt, _LinearSegmentedColormap, _PowerNorm = load_plotting()
        plt.rcParams.update({"font.family": "serif", "font.size": 12.5})
        fig, ax = plt.subplots(figsize=(9.6, 6.1), constrained_layout=True)
        held_color = "#c24b2a"
        for xpos in tick_positions:
            ax.axvline(xpos, color="#8a8a8a", linewidth=0.75, alpha=0.6, zorder=0)
        for band in range(held_thz.shape[1]):
            ax.plot(x_values, held_thz[:, band], color=held_color, linewidth=1.7, alpha=0.97)
        ax.axhline(0.0, color="black", linewidth=0.85, alpha=0.8)
        ax.set_xlim(float(x_values[0]), float(x_values[-1]))
        ax.set_xticks(tick_positions)
        ax.set_xticklabels(tick_labels)
        ax.set_ylabel("Frequency (THz)")
        ax.set_title(f"{phase.upper()} {metadata.symbol} HELD Mean Dispersion")
        ax.grid(axis="y", alpha=0.2)
        y_max = float(np.max(held_thz))
        y_min = float(np.min(held_thz))
        ax.set_ylim(min(0.0, y_min * 1.05), y_max * 1.05)
        output_plot = Path(output_plot)
        output_plot.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_plot, dpi=220)
        plt.close(fig)

    return {
        "held_thz": held_thz,
        "x_values": x_values,
        "tick_labels": tick_labels,
        "tick_positions": tick_positions,
        "q_path": q_path,
        "metadata": metadata,
    }


def x_edges_from_path(x_values: np.ndarray) -> np.ndarray:
    edges = np.empty(len(x_values) + 1, dtype=float)
    edges[1:-1] = 0.5 * (x_values[1:] + x_values[:-1])
    edges[0] = x_values[0] - (edges[1] - x_values[0])
    edges[-1] = x_values[-1] + (x_values[-1] - edges[-2])
    return edges


def build_intensity(
    x_values: np.ndarray,
    step_frequencies_thz: np.ndarray,
    y_bins: int,
    sigma_freq_thz: float,
    sigma_q_bins: float,
    y_min: float,
    y_max: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    x_samples = np.broadcast_to(x_values[None, :, None], step_frequencies_thz.shape).reshape(-1)
    y_samples = step_frequencies_thz.reshape(-1)
    x_edges = x_edges_from_path(x_values)
    y_edges = np.linspace(y_min, y_max, y_bins + 1, dtype=float)
    histogram, _, _ = np.histogram2d(x_samples, y_samples, bins=(x_edges, y_edges))
    intensity = gaussian_filter(histogram.T, sigma=(sigma_freq_thz / ((y_max - y_min) / y_bins), sigma_q_bins), mode="nearest")
    return intensity, x_edges, y_edges


def plot_heatmap(
    phase: str,
    npz_path: str | Path,
    held_csv: str | Path,
    cache_npz: str | Path | None = None,
    output_plot: str | Path | None = None,
    num_shells: int | None = None,
    cutoff_ang: float | None = None,
    mass_amu: float | None = None,
    path_labels: list[str] | None = None,
    points_per_segment: int = 90,
    y_bins: int = 900,
    sigma_freq_thz: float = 0.08,
    sigma_q_bins: float = 0.75,
    gamma: float = 0.42,
    vmax_percentile: float = 99.7,
    y_min: float | None = None,
    y_max: float | None = None,
    force_recompute: bool = False,
    verbose: bool = False,
) -> dict[str, object]:
    model, dataset, metadata = build_model_from_npz(
        phase=phase,
        npz_path=Path(npz_path),
        num_shells=num_shells,
        cutoff_ang=cutoff_ang,
        mass_amu=mass_amu,
    )
    _labels, _mean_values, step_values = read_fc_csv(Path(held_csv))
    q_path, x_values, tick_labels, tick_positions = build_q_path(
        phase=phase,
        primitive_cell=model.uc_cell,
        path_labels=path_labels,
        points_per_segment=points_per_segment,
    )

    cache_path = Path(cache_npz) if cache_npz is not None else None
    if cache_path is not None and cache_path.exists() and not force_recompute:
        cache = np.load(cache_path, allow_pickle=False)
        step_frequencies_thz = np.asarray(cache["step_frequencies_thz"], dtype=float)
    else:
        step_frequencies_thz = np.empty((step_values.shape[0], len(q_path), 3 * model.n_primitive), dtype=float)
        for step_index, coeffs in enumerate(step_values):
            if verbose and (step_index == 0 or (step_index + 1) % 25 == 0 or step_index + 1 == step_values.shape[0]):
                print(f"[HELD-HEATMAP] evaluating step {step_index + 1}/{step_values.shape[0]}")
            step_frequencies_thz[step_index] = model.dispersion_thz_from_reduced_path(coeffs, q_path)
        if cache_path is not None:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(cache_path, x_values=x_values, step_frequencies_thz=step_frequencies_thz)

    default_y_min = min(0.0, float(np.min(step_frequencies_thz)) - 1.0)
    default_y_max = float(np.max(step_frequencies_thz)) + 1.0
    final_y_min = default_y_min if y_min is None else float(y_min)
    final_y_max = default_y_max if y_max is None else float(y_max)
    intensity, _x_edges, y_edges = build_intensity(
        x_values,
        step_frequencies_thz,
        y_bins=y_bins,
        sigma_freq_thz=sigma_freq_thz,
        sigma_q_bins=sigma_q_bins,
        y_min=final_y_min,
        y_max=final_y_max,
    )

    if output_plot is not None:
        plt, LinearSegmentedColormap, PowerNorm = load_plotting()
        cmap = LinearSegmentedColormap.from_list(
            "black_red",
            ["#000000", "#160000", "#430000", "#8f0d0d", "#d73a1f", "#ff8a5a"],
        )
        vmax = float(np.percentile(intensity, vmax_percentile)) if np.max(intensity) > 0.0 else 1.0
        fig, ax = plt.subplots(figsize=(10.0, 6.2), constrained_layout=True)
        fig.patch.set_facecolor("black")
        ax.set_facecolor("black")
        ax.imshow(
            intensity,
            origin="lower",
            aspect="auto",
            cmap=cmap,
            norm=PowerNorm(gamma=gamma, vmin=0.0, vmax=max(vmax, 1.0)),
            extent=(x_values[0], x_values[-1], y_edges[0], y_edges[-1]),
            interpolation="bilinear",
        )
        for xpos in tick_positions:
            ax.axvline(xpos, color="#6b6b6b", linewidth=0.8, alpha=0.75, zorder=3)
        ax.set_xlim(float(x_values[0]), float(x_values[-1]))
        ax.set_ylim(final_y_min, final_y_max)
        ax.set_xticks(tick_positions)
        ax.set_xticklabels(tick_labels, color="white", fontsize=14)
        ax.set_ylabel("Frequency (THz)", color="white")
        ax.set_title(f"{phase.upper()} {metadata.symbol} HELD Heat Map", color="white", pad=12)
        ax.tick_params(axis="x", colors="white")
        ax.tick_params(axis="y", colors="white")
        for spine in ax.spines.values():
            spine.set_color("white")
        output_plot = Path(output_plot)
        output_plot.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_plot, dpi=240, facecolor=fig.get_facecolor())
        plt.close(fig)

    return {
        "step_frequencies_thz": step_frequencies_thz,
        "x_values": x_values,
        "tick_labels": tick_labels,
        "tick_positions": tick_positions,
        "metadata": metadata,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    def add_common(subparser: argparse.ArgumentParser, include_csv: bool = False) -> None:
        subparser.add_argument("--phase", choices=sorted(PHASES), required=True)
        subparser.add_argument("--npz", type=Path, required=True)
        subparser.add_argument("--num-shells", type=int, default=None)
        subparser.add_argument("--cutoff-ang", type=float, default=None)
        subparser.add_argument("--mass-amu", type=float, default=None)
        if include_csv:
            subparser.add_argument("--held-csv", type=Path, required=True)

    fit_parser = subparsers.add_parser("fit")
    add_common(fit_parser)
    fit_parser.add_argument("--output-csv", type=Path, required=True)
    fit_parser.add_argument("--aggregate", choices=["mean", "gaussian", "global"], default="mean")
    fit_parser.add_argument("--skip", type=int, default=0)
    fit_parser.add_argument("--every", type=int, default=1)
    fit_parser.add_argument("--max-frames", type=int, default=0)
    fit_parser.add_argument("--verbose", action="store_true")

    disp_parser = subparsers.add_parser("dispersion")
    add_common(disp_parser, include_csv=True)
    disp_parser.add_argument("--output-data", type=Path, required=True)
    disp_parser.add_argument("--output-plot", type=Path, required=True)
    disp_parser.add_argument("--path", default=None, help="Path like GM-X-W-K-GM-L")
    disp_parser.add_argument("--points-per-segment", type=int, default=90)

    heat_parser = subparsers.add_parser("heatmap")
    add_common(heat_parser, include_csv=True)
    heat_parser.add_argument("--cache-npz", type=Path, required=True)
    heat_parser.add_argument("--output-plot", type=Path, required=True)
    heat_parser.add_argument("--path", default=None, help="Path like GM-X-W-K-GM-L")
    heat_parser.add_argument("--points-per-segment", type=int, default=90)
    heat_parser.add_argument("--y-bins", type=int, default=900)
    heat_parser.add_argument("--sigma-freq-thz", type=float, default=0.08)
    heat_parser.add_argument("--sigma-q-bins", type=float, default=0.75)
    heat_parser.add_argument("--gamma", type=float, default=0.42)
    heat_parser.add_argument("--vmax-percentile", type=float, default=99.7)
    heat_parser.add_argument("--y-min", type=float, default=None)
    heat_parser.add_argument("--y-max", type=float, default=None)
    heat_parser.add_argument("--force-recompute", action="store_true")
    heat_parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()


def parse_path_labels(path: str | None) -> list[str] | None:
    if path is None:
        return None
    return [token.strip() for token in path.replace(" ", "").split("-") if token.strip()]


def main() -> int:
    args = parse_args()
    if args.command == "fit":
        result, info = fit_case(
            phase=args.phase,
            npz_path=args.npz,
            output_csv=args.output_csv,
            aggregate=args.aggregate,
            skip=args.skip,
            every=args.every,
            max_frames=args.max_frames,
            num_shells=args.num_shells,
            cutoff_ang=args.cutoff_ang,
            mass_amu=args.mass_amu,
            verbose=args.verbose,
        )
        print(
            f"[HELD] phase={info['phase']} symbol={info['symbol']} frames={info['n_frames']} "
            f"aggregate={args.aggregate} shells={info['num_shells']} -> {args.output_csv}"
        )
        print("[HELD] shell_distances_ang=" + ", ".join(f"{value:.6f}" for value in info["selected_shell_distances"]))
        print("[HELD] mean_coefficients=" + ", ".join(f"{label}={value:.6f}" for label, value in zip(result.labels, result.mean_values)))
        return 0

    if args.command == "dispersion":
        output = plot_mean_dispersion(
            phase=args.phase,
            npz_path=args.npz,
            held_csv=args.held_csv,
            output_data=args.output_data,
            output_plot=args.output_plot,
            num_shells=args.num_shells,
            cutoff_ang=args.cutoff_ang,
            mass_amu=args.mass_amu,
            path_labels=parse_path_labels(args.path),
            points_per_segment=args.points_per_segment,
        )
        held = output["held_thz"]
        print(f"[HELD] wrote {args.output_data}")
        print(f"[HELD] wrote {args.output_plot}")
        print(f"[HELD] min_freq_THz={np.min(held):.6f}")
        print(f"[HELD] max_freq_THz={np.max(held):.6f}")
        return 0

    if args.command == "heatmap":
        output = plot_heatmap(
            phase=args.phase,
            npz_path=args.npz,
            held_csv=args.held_csv,
            cache_npz=args.cache_npz,
            output_plot=args.output_plot,
            num_shells=args.num_shells,
            cutoff_ang=args.cutoff_ang,
            mass_amu=args.mass_amu,
            path_labels=parse_path_labels(args.path),
            points_per_segment=args.points_per_segment,
            y_bins=args.y_bins,
            sigma_freq_thz=args.sigma_freq_thz,
            sigma_q_bins=args.sigma_q_bins,
            gamma=args.gamma,
            vmax_percentile=args.vmax_percentile,
            y_min=args.y_min,
            y_max=args.y_max,
            force_recompute=args.force_recompute,
            verbose=args.verbose,
        )
        print(f"[HELD] wrote {args.cache_npz}")
        print(f"[HELD] wrote {args.output_plot}")
        print(f"[HELD] steps={output['step_frequencies_thz'].shape[0]} q_points={output['step_frequencies_thz'].shape[1]}")
        return 0

    raise RuntimeError(f"Unexpected command {args.command!r}")


if __name__ == "__main__":
    raise SystemExit(main())
