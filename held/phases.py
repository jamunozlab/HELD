from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from math import sqrt

import numpy as np


DISPLAY_LABELS = {"GM": "Γ"}


@dataclass(frozen=True)
class PhaseDefinition:
    name: str
    conventional_atoms: int
    default_shells: int
    default_path: tuple[str, ...]
    special_points: dict[str, np.ndarray]


PHASES = {
    "fcc": PhaseDefinition(
        name="fcc",
        conventional_atoms=4,
        default_shells=5,
        default_path=("GM", "X", "W", "K", "GM", "L"),
        special_points={
            "GM": np.array([0.0, 0.0, 0.0], dtype=float),
            "X": np.array([0.5, 0.0, 0.5], dtype=float),
            "W": np.array([0.5, 0.25, 0.75], dtype=float),
            "K": np.array([0.375, 0.375, 0.75], dtype=float),
            "L": np.array([0.5, 0.5, 0.5], dtype=float),
            "U": np.array([0.625, 0.25, 0.625], dtype=float),
        },
    ),
    "bcc": PhaseDefinition(
        name="bcc",
        conventional_atoms=2,
        default_shells=5,
        default_path=("GM", "H", "N", "GM", "P", "H"),
        special_points={
            "GM": np.array([0.0, 0.0, 0.0], dtype=float),
            "H": np.array([0.5, -0.5, 0.5], dtype=float),
            "N": np.array([0.0, 0.0, 0.5], dtype=float),
            "P": np.array([0.25, 0.25, 0.25], dtype=float),
        },
    ),
    "hcp": PhaseDefinition(
        name="hcp",
        conventional_atoms=2,
        default_shells=4,
        default_path=("GM", "M", "K", "GM", "A", "L", "H", "A"),
        special_points={
            "GM": np.array([0.0, 0.0, 0.0], dtype=float),
            "M": np.array([0.5, 0.0, 0.0], dtype=float),
            "K": np.array([1.0 / 3.0, 1.0 / 3.0, 0.0], dtype=float),
            "A": np.array([0.0, 0.0, 0.5], dtype=float),
            "L": np.array([0.5, 0.0, 0.5], dtype=float),
            "H": np.array([1.0 / 3.0, 1.0 / 3.0, 0.5], dtype=float),
        },
    ),
}


def factor_triplets(total: int) -> list[tuple[int, int, int]]:
    triplets: list[tuple[int, int, int]] = []
    for nx in range(1, total + 1):
        if total % nx != 0:
            continue
        rem = total // nx
        for ny in range(1, rem + 1):
            if rem % ny != 0:
                continue
            nz = rem // ny
            triplets.append((nx, ny, nz))
    return triplets


def infer_cubic_repetitions(cell_ang: np.ndarray, natoms: int, conventional_atoms: int) -> tuple[int, int, int, float]:
    nconv = natoms // conventional_atoms
    lengths = np.linalg.norm(np.asarray(cell_ang, dtype=float), axis=1)
    candidates: list[tuple[float, tuple[int, int, int], float]] = []
    for reps in factor_triplets(nconv):
        reps_arr = np.array(reps, dtype=float)
        a_guess = float(np.mean(lengths / reps_arr))
        mismatch = float(np.max(np.abs(lengths - a_guess * reps_arr)))
        anis = float(abs(reps[0] - reps[1]) + abs(reps[1] - reps[2]))
        candidates.append((mismatch + 1.0e-9 * anis, reps, a_guess))
    candidates.sort(key=lambda item: item[0])
    best = candidates[0]
    return best[1][0], best[1][1], best[1][2], best[2]


def infer_hexagonal_repetitions(cell_ang: np.ndarray, natoms: int, conventional_atoms: int) -> tuple[int, int, int, float, float]:
    nconv = natoms // conventional_atoms
    cell_ang = np.asarray(cell_ang, dtype=float)
    sign_x = 1.0 if cell_ang[1, 0] >= 0.0 else -1.0
    candidates: list[tuple[float, tuple[int, int, int], float, float]] = []
    for nx, ny, nz in factor_triplets(nconv):
        a0 = np.linalg.norm(cell_ang[0]) / nx
        a1 = np.linalg.norm(cell_ang[1]) / ny
        a_guess = 0.5 * (a0 + a1)
        c_guess = np.linalg.norm(cell_ang[2]) / nz
        expected = np.array(
            [
                [a_guess * nx, 0.0, 0.0],
                [sign_x * 0.5 * a_guess * ny, 0.5 * sqrt(3.0) * a_guess * ny, 0.0],
                [0.0, 0.0, c_guess * nz],
            ],
            dtype=float,
        )
        mismatch = float(np.max(np.abs(cell_ang - expected)))
        anis = float(abs(nx - ny) + abs(ny - nz) + abs(nx - nz))
        candidates.append((mismatch + 1.0e-9 * anis, (nx, ny, nz), a_guess, c_guess))
    candidates.sort(key=lambda item: item[0])
    best = candidates[0]
    return best[1][0], best[1][1], best[1][2], best[2], best[3]


def primitive_cell_and_basis(phase: str, cell_ang: np.ndarray, natoms: int) -> tuple[np.ndarray, np.ndarray]:
    phase = phase.lower()
    if phase == "fcc":
        _nx, _ny, _nz, lattice_a = infer_cubic_repetitions(cell_ang, natoms, PHASES[phase].conventional_atoms)
        primitive = np.array(
            [
                [0.0, 0.5 * lattice_a, 0.5 * lattice_a],
                [0.5 * lattice_a, 0.0, 0.5 * lattice_a],
                [0.5 * lattice_a, 0.5 * lattice_a, 0.0],
            ],
            dtype=float,
        )
        basis = np.array([[0.0, 0.0, 0.0]], dtype=float)
        return primitive, basis
    if phase == "bcc":
        _nx, _ny, _nz, lattice_a = infer_cubic_repetitions(cell_ang, natoms, PHASES[phase].conventional_atoms)
        primitive = np.array(
            [
                [-0.5 * lattice_a, 0.5 * lattice_a, 0.5 * lattice_a],
                [0.5 * lattice_a, -0.5 * lattice_a, 0.5 * lattice_a],
                [0.5 * lattice_a, 0.5 * lattice_a, -0.5 * lattice_a],
            ],
            dtype=float,
        )
        basis = np.array([[0.0, 0.0, 0.0]], dtype=float)
        return primitive, basis
    if phase == "hcp":
        _nx, _ny, _nz, lattice_a, lattice_c = infer_hexagonal_repetitions(cell_ang, natoms, PHASES[phase].conventional_atoms)
        sign_x = 1.0 if np.asarray(cell_ang, dtype=float)[1, 0] >= 0.0 else -1.0
        primitive = np.array(
            [
                [lattice_a, 0.0, 0.0],
                [sign_x * 0.5 * lattice_a, 0.5 * sqrt(3.0) * lattice_a, 0.0],
                [0.0, 0.0, lattice_c],
            ],
            dtype=float,
        )
        basis = np.array([[0.0, 0.0, 0.0], [2.0 / 3.0, 1.0 / 3.0, 0.5]], dtype=float)
        return primitive, basis
    raise ValueError(f"Unsupported phase {phase!r}.")


def format_tick_label(label: str) -> str:
    return DISPLAY_LABELS.get(label, label)


def build_q_path(
    phase: str,
    primitive_cell: np.ndarray,
    path_labels: list[str] | tuple[str, ...] | None = None,
    points_per_segment: int = 90,
) -> tuple[np.ndarray, np.ndarray, list[str], np.ndarray]:
    phase_def = PHASES[phase.lower()]
    labels = list(phase_def.default_path if path_labels is None else path_labels)
    if len(labels) < 2:
        raise ValueError("At least two path labels are required.")

    special_points = phase_def.special_points
    for label in labels:
        if label not in special_points:
            raise ValueError(f"Unsupported special-point label {label!r} for phase {phase}.")

    q_blocks: list[np.ndarray] = []
    tick_positions: list[float] = [0.0]
    reciprocal = 2.0 * np.pi * np.linalg.inv(np.asarray(primitive_cell, dtype=float)).T
    cumulative = 0.0

    for segment_index, (label_a, label_b) in enumerate(zip(labels[:-1], labels[1:])):
        start = special_points[label_a]
        end = special_points[label_b]
        steps = np.linspace(0.0, 1.0, points_per_segment + 1, dtype=float)
        if segment_index > 0:
            steps = steps[1:]
        segment = start[None, :] + steps[:, None] * (end - start)[None, :]
        q_blocks.append(segment)
        seg_cart = segment @ reciprocal.T
        if segment_index == 0:
            distances = np.linalg.norm(np.diff(seg_cart, axis=0), axis=1)
        else:
            distances = np.linalg.norm(np.diff(seg_cart, axis=0), axis=1)
        cumulative += float(np.sum(distances))
        tick_positions.append(cumulative)

    q_path = np.vstack(q_blocks)
    q_cart = q_path @ reciprocal.T
    x_values = np.zeros(len(q_path), dtype=float)
    if len(q_path) > 1:
        x_values[1:] = np.cumsum(np.linalg.norm(np.diff(q_cart, axis=0), axis=1))
    return q_path, x_values, [format_tick_label(label) for label in labels], np.array(tick_positions, dtype=float)
