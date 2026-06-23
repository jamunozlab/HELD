from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np


RY_TO_EV = 13.605693122994
BOHR_TO_ANG = 0.529177210903
RY_PER_BOHR_TO_EV_PER_ANG = RY_TO_EV / BOHR_TO_ANG


@dataclass
class HeldRunResult:
    step_ids: list[int]
    labels: list[str]
    mean_values: np.ndarray
    fitted_sigmas: np.ndarray
    step_values: np.ndarray

    def as_dict(self) -> dict[str, float]:
        return {label: float(value) for label, value in zip(self.labels, self.mean_values)}


@dataclass
class TrajectoryDataset:
    path: Path
    symbols: list[str]
    ideal_cell_ang: np.ndarray
    ideal_frac: np.ndarray
    positions_frac: np.ndarray
    forces_ev_ang: np.ndarray
    step_ids: np.ndarray

    @property
    def natoms(self) -> int:
        return int(self.ideal_frac.shape[0])

    @property
    def unique_symbols(self) -> list[str]:
        return sorted(set(self.symbols))

    def select_frames(
        self,
        skip: int = 0,
        every: int = 1,
        max_frames: int = 0,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        finite_mask = np.isfinite(self.positions_frac).all(axis=(1, 2)) & np.isfinite(self.forces_ev_ang).all(axis=(1, 2))
        selected = np.arange(len(self.positions_frac))[skip::every]
        selected = selected[finite_mask[selected]]
        if max_frames > 0:
            selected = selected[:max_frames]
        if len(selected) == 0:
            raise ValueError(f"No finite frames selected from {self.path}.")
        return self.positions_frac[selected], self.forces_ev_ang[selected], self.step_ids[selected]


def wrap_fractional(frac: np.ndarray) -> np.ndarray:
    wrapped = np.mod(np.asarray(frac, dtype=float), 1.0)
    wrapped[np.isclose(wrapped, 1.0, atol=1.0e-8)] = 0.0
    return wrapped


def load_npz_dataset(path: Path) -> TrajectoryDataset:
    path = Path(path)
    data = np.load(path, allow_pickle=False)
    input_cell = np.asarray(data["input_cell_parameters"], dtype=float)
    input_unit = str(np.asarray(data["input_cell_unit"]).item()).lower()
    if input_unit not in {"angstrom", "ang"}:
        raise ValueError(f"Unsupported input_cell_unit={input_unit!r} in {path}.")

    positions_frac = np.asarray(data["positions"], dtype=float)
    positions_units = {str(value).lower() for value in np.asarray(data["positions_unit"])}
    if positions_units != {"crystal"}:
        raise ValueError(f"Unsupported positions_unit values in {path}: {sorted(positions_units)}")

    ideal_frac = np.asarray(data["initial_positions_alat"], dtype=float) @ np.linalg.inv(
        np.asarray(data["initial_cell_alat"], dtype=float)
    )
    ideal_frac = wrap_fractional(ideal_frac)
    forces_ev_ang = np.asarray(data["forces_ry_au"], dtype=float) * RY_PER_BOHR_TO_EV_PER_ANG
    step_ids = np.asarray(data["iteration"], dtype=int) if "iteration" in data.files else np.arange(len(positions_frac), dtype=int)
    symbols = [str(symbol) for symbol in np.asarray(data["symbols"]).tolist()]

    return TrajectoryDataset(
        path=path,
        symbols=symbols,
        ideal_cell_ang=input_cell,
        ideal_frac=ideal_frac,
        positions_frac=positions_frac,
        forces_ev_ang=forces_ev_ang,
        step_ids=step_ids,
    )


def read_fc_csv(path: Path) -> tuple[list[str], np.ndarray, np.ndarray]:
    lines = Path(path).read_text().splitlines()
    if len(lines) < 3:
        raise ValueError(f"{path} is too short to be a HELD CSV file.")

    labels = [token.strip() for token in lines[0].split(",") if token.strip()]
    averages = np.array([float(token.strip()) for token in lines[2].split(",") if token.strip()], dtype=float)
    step_rows = [
        np.array([float(token.strip()) for token in line.split(",") if token.strip()], dtype=float)
        for line in lines[4:]
        if line.strip()
    ]
    return labels, averages, np.array(step_rows, dtype=float)


def write_fc_csv(path: Path, result: HeldRunResult) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as handle:
        for label in result.labels:
            print(f"{label}, ", end="", file=handle)
        print("\n", file=handle)

        for value in result.mean_values:
            print(f"{value}, ", end="", file=handle)
        print("\n", file=handle)

        for row in result.step_values:
            for value in row:
                print(f"{value}, ", end="", file=handle)
            print("", file=handle)
