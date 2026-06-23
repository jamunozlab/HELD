from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from pathlib import Path

import numpy as np
import spglib
from ase.data import atomic_masses, atomic_numbers

from .io import HeldRunResult, TrajectoryDataset, load_npz_dataset
from .phases import PHASES, primitive_cell_and_basis


EV_TO_J = 1.602176634e-19
AMU_TO_KG = 1.66053906660e-27
ANGSTROM_TO_M = 1.0e-10
OMEGA2_CONVERSION = EV_TO_J / (AMU_TO_KG * ANGSTROM_TO_M**2)
THZ_TO_MEV = 4.135667696


def gaussian_fit(values: np.ndarray) -> tuple[float, float]:
    array = np.asarray(values, dtype=float)
    return float(np.mean(array)), float(np.std(array, ddof=0))


def matrix_null_space(matrix: np.ndarray, rtol: float = 1.0e-10) -> np.ndarray:
    matrix = np.asarray(matrix, dtype=float)
    if matrix.size == 0:
        return np.eye(matrix.shape[1], dtype=float)
    gram = matrix.T @ matrix
    eigenvalues, eigenvectors = np.linalg.eigh(gram)
    max_eval = float(np.max(np.abs(eigenvalues))) if eigenvalues.size else 0.0
    cutoff = rtol * max_eval
    keep = np.abs(eigenvalues) <= cutoff
    return np.array(eigenvectors[:, keep], dtype=float, copy=True)


@dataclass
class ModelMetadata:
    phase: str
    symbol: str
    mass_amu: float
    num_shells: int
    selected_shell_distances: list[float]


class SymmetryHeldModel:
    def __init__(
        self,
        primitive_cell: np.ndarray,
        basis_frac: np.ndarray,
        supercell_cell: np.ndarray,
        ideal_supercell_frac: np.ndarray,
        atomic_number: int,
        mass_amu: float,
        num_shells: int,
        cutoff_ang: float | None = None,
        symprec: float = 1.0e-5,
    ) -> None:
        self.uc_cell = np.asarray(primitive_cell, dtype=float)
        self.uc_frac = np.asarray(basis_frac, dtype=float)
        self.ss_cell = np.asarray(supercell_cell, dtype=float)
        self.ss_frac = np.mod(np.asarray(ideal_supercell_frac, dtype=float), 1.0)
        self.atomic_number = int(atomic_number)
        self.mass_amu = float(mass_amu)
        self.num_shells = int(num_shells)
        self.cutoff_ang = cutoff_ang
        self.symprec = float(symprec)
        self.mapping_tol = max(1.0e-5, 10.0 * self.symprec)

        self.n_primitive = len(self.uc_frac)
        self.n_supercell = len(self.ss_frac)
        self.supercell_transform = self._infer_supercell_transform()
        self.ideal_total_coords, self.supercell_basis_ids, self.supercell_cell_ids = self._map_supercell_to_primitive()
        self.offsite_keys, self.selected_shell_distances = self._generate_shell_pairs()
        self.offsite_index = {key: index for index, key in enumerate(self.offsite_keys)}
        self.full_pair_key_map = self._build_supercell_pair_map()
        self.basis_vectors = self._build_symmetry_basis()
        self.basis_labels = [f"b{index:03d}" for index in range(self.basis_vectors.shape[1])]
        self.basis_offsite_mats, self.basis_onsite_mats = self._basis_matrices()
        self.pair_a, self.pair_b, self.pair_key_index = self._pair_arrays()
        self.pair_basis_mats = np.transpose(self.basis_offsite_mats[:, self.pair_key_index], (1, 0, 2, 3))
        self.offsite_atom_i = np.array([key[0] for key in self.offsite_keys], dtype=int)
        self.offsite_atom_j = np.array([key[1] for key in self.offsite_keys], dtype=int)
        self.offsite_separations_cart = np.array(
            [
                (self.uc_frac[key[1]] + np.array(key[2], dtype=float) - self.uc_frac[key[0]]) @ self.uc_cell
                for key in self.offsite_keys
            ],
            dtype=float,
        )

    def _infer_supercell_transform(self) -> np.ndarray:
        transform = self.ss_cell @ np.linalg.inv(self.uc_cell)
        rounded = np.rint(transform).astype(int)
        if not np.allclose(transform, rounded, atol=self.mapping_tol):
            raise ValueError(f"Supercell transform is not integer-valued:\n{transform}")
        expected_cells = self.n_supercell // self.n_primitive
        det = int(round(abs(np.linalg.det(rounded))))
        if det != expected_cells:
            raise ValueError(f"Transform determinant {det} does not match expected cell count {expected_cells}.")
        return rounded

    def _map_supercell_to_primitive(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        supercell_cart = self.ss_frac @ self.ss_cell
        primitive_total = supercell_cart @ np.linalg.inv(self.uc_cell)
        basis_ids: list[int] = []
        cell_ids: list[tuple[int, int, int]] = []
        for frac_total in primitive_total:
            found = False
            for basis_index, basis_frac in enumerate(self.uc_frac):
                diff = frac_total - basis_frac
                shift = np.rint(diff).astype(int)
                if np.allclose(diff - shift, 0.0, atol=self.mapping_tol):
                    basis_ids.append(basis_index)
                    cell_ids.append(tuple(int(value) for value in shift))
                    found = True
                    break
            if not found:
                raise ValueError(f"Could not map supercell atom at primitive coordinate {frac_total.tolist()}.")
        return primitive_total, np.array(basis_ids, dtype=int), np.array(cell_ids, dtype=int)

    def _enumerate_candidates(self, search_range: int) -> list[tuple[tuple[int, int, tuple[int, int, int]], float]]:
        candidates: list[tuple[tuple[int, int, tuple[int, int, int]], float]] = []
        for atom_i in range(self.n_primitive):
            for atom_j in range(self.n_primitive):
                for shift in product(range(-search_range, search_range + 1), repeat=3):
                    if atom_i == atom_j and shift == (0, 0, 0):
                        continue
                    separation = (self.uc_frac[atom_j] + np.array(shift, dtype=float) - self.uc_frac[atom_i]) @ self.uc_cell
                    distance = float(np.linalg.norm(separation))
                    if distance <= 1.0e-10:
                        continue
                    candidates.append(((atom_i, atom_j, shift), distance))
        candidates.sort(key=lambda item: (item[1], item[0][0], item[0][1], item[0][2]))
        return candidates

    def _group_distances(self, distances: list[float], tol: float) -> list[float]:
        unique: list[float] = []
        for distance in distances:
            if not unique or abs(distance - unique[-1]) > tol:
                unique.append(distance)
        return unique

    def _generate_shell_pairs(self) -> tuple[list[tuple[int, int, tuple[int, int, int]]], list[float]]:
        search_range = 1
        tol = 1.0e-5 * min(np.linalg.norm(self.uc_cell[0]), np.linalg.norm(self.uc_cell[1]), np.linalg.norm(self.uc_cell[2]))
        while True:
            candidates = self._enumerate_candidates(search_range)
            unique_distances = self._group_distances([distance for _key, distance in candidates], tol)
            if self.cutoff_ang is not None:
                enough = unique_distances and unique_distances[-1] >= self.cutoff_ang - tol
            else:
                enough = len(unique_distances) >= self.num_shells
            if enough or search_range >= 8:
                break
            search_range += 1

        if self.cutoff_ang is not None:
            selected = [distance for distance in unique_distances if distance <= self.cutoff_ang + tol]
        else:
            selected = unique_distances[: self.num_shells]
        keys = [
            key
            for key, distance in candidates
            if any(abs(distance - shell_distance) <= tol for shell_distance in selected)
        ]
        return keys, [float(value) for value in selected]

    def _supercell_lattice_shifts(self) -> list[np.ndarray]:
        return [np.array(multipliers, dtype=int) @ self.supercell_transform for multipliers in product((-1, 0, 1), repeat=3)]

    def _build_supercell_pair_map(self) -> dict[tuple[int, int], tuple[int, int, tuple[int, int, int]]]:
        mapping: dict[tuple[int, int], tuple[int, int, tuple[int, int, int]]] = {}
        lattice_shifts = self._supercell_lattice_shifts()
        for atom_a in range(self.n_supercell):
            basis_a = int(self.supercell_basis_ids[atom_a])
            cell_a = self.supercell_cell_ids[atom_a]
            for atom_b in range(self.n_supercell):
                if atom_a == atom_b:
                    continue
                basis_b = int(self.supercell_basis_ids[atom_b])
                delta = self.supercell_cell_ids[atom_b] - cell_a
                found_key = None
                for lattice_shift in lattice_shifts:
                    key = (basis_a, basis_b, tuple((delta + lattice_shift).tolist()))
                    if key in self.offsite_index:
                        found_key = key
                        break
                if found_key is not None:
                    mapping[(atom_a, atom_b)] = found_key
        return mapping

    def _primitive_symmetry(self) -> tuple[np.ndarray, np.ndarray]:
        numbers = [self.atomic_number] * self.n_primitive
        symmetry = spglib.get_symmetry((self.uc_cell, self.uc_frac, numbers), symprec=self.symprec)
        return np.asarray(symmetry["rotations"], dtype=int), np.asarray(symmetry["translations"], dtype=float)

    def _map_primitive_atom(self, frac_coord: np.ndarray, rotation: np.ndarray, translation: np.ndarray) -> tuple[int, np.ndarray]:
        transformed = rotation @ frac_coord + translation
        for atom_j, basis_frac in enumerate(self.uc_frac):
            diff = transformed - basis_frac
            shift = np.rint(diff).astype(int)
            if np.allclose(diff - shift, 0.0, atol=self.mapping_tol):
                return atom_j, shift
        raise ValueError(f"Could not map primitive atom at {frac_coord.tolist()} through symmetry.")

    def _build_symmetry_basis(self) -> np.ndarray:
        rotations, translations = self._primitive_symmetry()
        n_offsite = len(self.offsite_keys)
        elementary = []
        for axis_a in range(3):
            for axis_b in range(3):
                mat = np.zeros((3, 3), dtype=float)
                mat[axis_a, axis_b] = 1.0
                elementary.append(mat)

        rows = []
        for rotation, translation in zip(rotations, translations):
            cart_rotation = self.uc_cell.T @ rotation @ np.linalg.inv(self.uc_cell.T)
            atom_map = [self._map_primitive_atom(frac, rotation, translation) for frac in self.uc_frac]
            transform = np.column_stack([(cart_rotation @ basis @ cart_rotation.T).reshape(-1) for basis in elementary])
            for pair_key in self.offsite_keys:
                atom_i, atom_j, lattice_vector = pair_key
                mapped_i, shift_i = atom_map[atom_i]
                mapped_j, shift_j = self._map_primitive_atom(
                    self.uc_frac[atom_j] + np.array(lattice_vector, dtype=float),
                    rotation,
                    translation,
                )
                mapped_key = (mapped_i, mapped_j, tuple((shift_j - shift_i).tolist()))
                row_pair = np.zeros((9, 9 * n_offsite), dtype=float)
                row_pair[:, 9 * self.offsite_index[mapped_key] : 9 * (self.offsite_index[mapped_key] + 1)] = np.eye(9)
                row_pair[:, 9 * self.offsite_index[pair_key] : 9 * (self.offsite_index[pair_key] + 1)] -= transform
                rows.append(row_pair)

        for pair_key in self.offsite_keys:
            atom_i, atom_j, lattice_vector = pair_key
            reverse_key = (atom_j, atom_i, tuple((-np.array(lattice_vector)).tolist()))
            left = self.offsite_index[pair_key]
            right = self.offsite_index[reverse_key]
            for axis_a in range(3):
                for axis_b in range(3):
                    row = np.zeros(9 * n_offsite, dtype=float)
                    row[9 * right + 3 * axis_a + axis_b] = 1.0
                    row[9 * left + 3 * axis_b + axis_a] -= 1.0
                    rows.append(row[None, :])

        constraint_matrix = np.concatenate(rows, axis=0)
        return matrix_null_space(constraint_matrix)

    def _onsite_from_offsite(self, offsite_mats: np.ndarray) -> np.ndarray:
        offsite_mats = np.asarray(offsite_mats, dtype=float)
        if offsite_mats.ndim == 4:
            onsite = np.zeros((offsite_mats.shape[0], self.n_primitive, 3, 3), dtype=float)
            for pair_index, pair_key in enumerate(self.offsite_keys):
                onsite[:, pair_key[0]] -= offsite_mats[:, pair_index]
            return onsite
        onsite = np.zeros((self.n_primitive, 3, 3), dtype=float)
        for pair_index, pair_key in enumerate(self.offsite_keys):
            onsite[pair_key[0]] -= offsite_mats[pair_index]
        return onsite

    def _basis_matrices(self) -> tuple[np.ndarray, np.ndarray]:
        n_basis = self.basis_vectors.shape[1]
        n_offsite = len(self.offsite_keys)
        offsite = self.basis_vectors.T.reshape(n_basis, n_offsite, 3, 3)
        onsite = self._onsite_from_offsite(offsite)
        return offsite, onsite

    def _pair_arrays(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        pair_a: list[int] = []
        pair_b: list[int] = []
        pair_key_index: list[int] = []
        for (atom_a, atom_b), pair_key in self.full_pair_key_map.items():
            pair_a.append(atom_a)
            pair_b.append(atom_b)
            pair_key_index.append(self.offsite_index[pair_key])
        return np.array(pair_a, dtype=int), np.array(pair_b, dtype=int), np.array(pair_key_index, dtype=int)

    def frame_displacements_cart(self, frame_frac_super: np.ndarray) -> np.ndarray:
        delta_frac = np.asarray(frame_frac_super, dtype=float) - self.ss_frac
        delta_frac -= np.rint(delta_frac)
        return delta_frac @ self.ss_cell

    def build_design_matrix(self, displacements_cart: np.ndarray) -> np.ndarray:
        du_pairs = np.asarray(displacements_cart[self.pair_a] - displacements_cart[self.pair_b], dtype=float)
        pair_contrib = np.einsum("pmij,pj->pmi", self.pair_basis_mats, du_pairs, optimize=True)
        design = np.zeros((3 * self.n_supercell, self.basis_vectors.shape[1]), dtype=float)
        for basis_index in range(self.basis_vectors.shape[1]):
            forces = np.zeros((self.n_supercell, 3), dtype=float)
            np.add.at(forces, self.pair_a, pair_contrib[:, basis_index, :])
            design[:, basis_index] = forces.reshape(-1)
        return design

    def solve_step(self, frame_frac_super: np.ndarray, forces_ev_ang: np.ndarray) -> np.ndarray:
        displacements_cart = self.frame_displacements_cart(frame_frac_super)
        design = self.build_design_matrix(displacements_cart)
        coeffs, *_ = np.linalg.lstsq(design, np.asarray(forces_ev_ang, dtype=float).reshape(-1), rcond=None)
        return coeffs

    def solve_global_series(
        self,
        positions_frac: np.ndarray,
        forces_ev_ang: np.ndarray,
        step_ids: np.ndarray | None = None,
        verbose: bool = False,
    ) -> np.ndarray:
        design_blocks: list[np.ndarray] = []
        force_blocks: list[np.ndarray] = []
        for index in range(len(positions_frac)):
            displacements_cart = self.frame_displacements_cart(positions_frac[index])
            design_blocks.append(self.build_design_matrix(displacements_cart))
            force_blocks.append(np.asarray(forces_ev_ang[index], dtype=float).reshape(-1))
            if verbose and (index == 0 or (index + 1) % 25 == 0 or index + 1 == len(positions_frac)):
                label = f"step={int(step_ids[index])}" if step_ids is not None else f"frame={index + 1}"
                print(f"[HELD] assembled global frame {index + 1}/{len(positions_frac)} ({label})")

        global_design = np.vstack(design_blocks)
        global_forces = np.concatenate(force_blocks)
        coeffs, *_ = np.linalg.lstsq(global_design, global_forces, rcond=None)
        return coeffs

    def fit_series(
        self,
        positions_frac: np.ndarray,
        forces_ev_ang: np.ndarray,
        step_ids: np.ndarray,
        aggregate: str = "mean",
        verbose: bool = False,
    ) -> HeldRunResult:
        positions_frac = np.asarray(positions_frac, dtype=float)
        forces_ev_ang = np.asarray(forces_ev_ang, dtype=float)
        step_ids = np.asarray(step_ids, dtype=int)
        step_values = np.zeros((positions_frac.shape[0], self.basis_vectors.shape[1]), dtype=float)
        for index in range(positions_frac.shape[0]):
            step_values[index] = self.solve_step(positions_frac[index], forces_ev_ang[index])
            if verbose and (index == 0 or (index + 1) % 25 == 0 or index + 1 == positions_frac.shape[0]):
                print(f"[HELD] solved frame {index + 1}/{positions_frac.shape[0]} (step={int(step_ids[index])})")

        sigmas = np.zeros(self.basis_vectors.shape[1], dtype=float)
        for column in range(self.basis_vectors.shape[1]):
            _mu, sigmas[column] = gaussian_fit(step_values[:, column])

        aggregate = aggregate.lower()
        if aggregate == "mean":
            mean_values = step_values.mean(axis=0)
        elif aggregate == "gaussian":
            mean_values = np.array([gaussian_fit(step_values[:, column])[0] for column in range(step_values.shape[1])], dtype=float)
        elif aggregate == "global":
            mean_values = self.solve_global_series(positions_frac, forces_ev_ang, step_ids=step_ids, verbose=verbose)
        else:
            raise ValueError(f"Unsupported aggregate mode {aggregate!r}. Use mean, gaussian, or global.")

        return HeldRunResult(
            step_ids=[int(step) for step in step_ids.tolist()],
            labels=self.basis_labels[:],
            mean_values=np.asarray(mean_values, dtype=float),
            fitted_sigmas=sigmas,
            step_values=step_values,
        )

    def primitive_force_constants_from_coefficients(self, coeffs: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        coeffs = np.asarray(coeffs, dtype=float)
        offsite = np.tensordot(coeffs, self.basis_offsite_mats, axes=(0, 0))
        onsite = np.tensordot(coeffs, self.basis_onsite_mats, axes=(0, 0))
        return offsite, onsite

    def _mass_factor(self, atom_i: int, atom_j: int) -> float:
        return float(np.sqrt(self.mass_amu * self.mass_amu))

    def q_cart_cycles_from_reduced(self, q_reduced: np.ndarray) -> np.ndarray:
        q_reduced = np.asarray(q_reduced, dtype=float)
        return q_reduced @ np.linalg.inv(self.uc_cell)

    def dispersion_thz_from_reduced_path(self, coeffs: np.ndarray, q_reduced_path: np.ndarray) -> np.ndarray:
        return self.dispersion_thz_from_cart_cycles(coeffs, self.q_cart_cycles_from_reduced(q_reduced_path))

    def dispersion_thz_from_cart_cycles(self, coeffs: np.ndarray, q_cart_cycles: np.ndarray) -> np.ndarray:
        offsite, onsite = self.primitive_force_constants_from_coefficients(coeffs)
        q_phase = 2.0 * np.pi * np.asarray(q_cart_cycles, dtype=float)
        phases = np.exp(1j * q_phase @ self.offsite_separations_cart.T)
        n_q = q_phase.shape[0]
        n_mode = 3 * self.n_primitive
        dyn = np.zeros((n_q, n_mode, n_mode), dtype=complex)

        for atom_i in range(self.n_primitive):
            row = slice(3 * atom_i, 3 * (atom_i + 1))
            dyn[:, row, row] += onsite[atom_i][None, :, :] / self._mass_factor(atom_i, atom_i)

        for pair_index in range(len(self.offsite_keys)):
            atom_i = int(self.offsite_atom_i[pair_index])
            atom_j = int(self.offsite_atom_j[pair_index])
            row = slice(3 * atom_i, 3 * (atom_i + 1))
            col = slice(3 * atom_j, 3 * (atom_j + 1))
            dyn[:, row, col] += phases[:, pair_index][:, None, None] * (
                offsite[pair_index][None, :, :] / self._mass_factor(atom_i, atom_j)
            )

        dyn = 0.5 * (dyn + np.conjugate(np.swapaxes(dyn, 1, 2)))
        eigenvalues = np.linalg.eigvalsh(dyn)
        freqs_thz = np.zeros_like(eigenvalues, dtype=float)
        positive = np.clip(eigenvalues, 0.0, None)
        negative = np.clip(-eigenvalues, 0.0, None)
        freqs_thz += np.sqrt(positive * OMEGA2_CONVERSION) / (2.0 * np.pi) * 1.0e-12
        freqs_thz -= np.sqrt(negative * OMEGA2_CONVERSION) / (2.0 * np.pi) * 1.0e-12
        return freqs_thz


def build_model_from_npz(
    phase: str,
    npz_path: Path,
    num_shells: int | None = None,
    cutoff_ang: float | None = None,
    mass_amu: float | None = None,
) -> tuple[SymmetryHeldModel, TrajectoryDataset, ModelMetadata]:
    dataset = load_npz_dataset(npz_path)
    unique_symbols = dataset.unique_symbols
    if len(unique_symbols) != 1:
        raise ValueError(f"Current unified HELD implementation expects a single-element dataset, got {unique_symbols}.")
    symbol = unique_symbols[0]
    atomic_number = int(atomic_numbers[symbol])
    mass = float(mass_amu if mass_amu is not None else atomic_masses[atomic_number])
    phase_def = PHASES[phase.lower()]
    shell_count = phase_def.default_shells if num_shells is None else int(num_shells)
    primitive_cell, basis_frac = primitive_cell_and_basis(phase, dataset.ideal_cell_ang, dataset.natoms)
    model = SymmetryHeldModel(
        primitive_cell=primitive_cell,
        basis_frac=basis_frac,
        supercell_cell=dataset.ideal_cell_ang,
        ideal_supercell_frac=dataset.ideal_frac,
        atomic_number=atomic_number,
        mass_amu=mass,
        num_shells=shell_count,
        cutoff_ang=cutoff_ang,
    )
    metadata = ModelMetadata(
        phase=phase.lower(),
        symbol=symbol,
        mass_amu=mass,
        num_shells=shell_count,
        selected_shell_distances=model.selected_shell_distances,
    )
    return model, dataset, metadata
