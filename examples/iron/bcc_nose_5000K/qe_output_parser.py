import argparse
import os
import re
import json
import lzma
import pickle
from pathlib import Path

import numpy as np

# ============================================================
# USER SETTINGS
# ============================================================

ROOT_DIR = '/scratch/dajuarez4/testing/non-magnetic'
OUTPUT_DIR = '/scratch/dajuarez4/testing/non-magnetic/'

SAVE_FMT = "npz"   # "npz" or "pkl_xz"
OUT_FILE_EXTENSIONS = (".out",)
CHOOSE_IF_MULTIPLE = "largest"   # "largest" or "newest"
SKIP_EMPTY = True

# ============================================================
# CONSTANTS
# ============================================================

BOHR_TO_ANG = 0.529177210903
RY_TO_EV = 13.605693009
RY_BOHR_TO_EV_ANG = RY_TO_EV / BOHR_TO_ANG
KBAR_TO_GPA = 0.1

FLOAT_RE = r"[-+]?(?:\d+\.\d*|\.\d+|\d+)(?:[EeDd][-+]?\d+)?"

# ============================================================
# REGEX
# ============================================================

re_natoms = re.compile(r"number of atoms/cell\s*=\s*(\d+)")
re_alat = re.compile(r"lattice parameter \(alat\)\s*=\s*(" + FLOAT_RE + r")\s*a\.u\.")
re_ax = re.compile(r"a\(\d\)\s*=\s*\(\s*(" + FLOAT_RE + r")\s+(" + FLOAT_RE + r")\s+(" + FLOAT_RE + r")\s*\)")
re_enter_dyn = re.compile(r"Entering Dynamics:\s*iteration\s*=\s*(\d+)")
re_time_ps = re.compile(r"time\s*=\s*(" + FLOAT_RE + r")\s*pico-seconds")
re_total_energy = re.compile(r"!\s*total energy\s*=\s*(" + FLOAT_RE + r")\s*Ry")
re_internal_energy = re.compile(r"internal energy E=F\+TS\s*=\s*(" + FLOAT_RE + r")\s*Ry")
re_temperature = re.compile(r"temperature\s*=\s*(" + FLOAT_RE + r")\s*K")
re_ekin = re.compile(r"kinetic energy \(Ekin\)\s*=\s*(" + FLOAT_RE + r")\s*Ry")
re_pressure = re.compile(r"P=\s*(" + FLOAT_RE + r")")
re_total_mag = re.compile(r"total magnetization\s*=\s*(" + FLOAT_RE + r")")
re_abs_mag = re.compile(r"absolute magnetization\s*=\s*(" + FLOAT_RE + r")")
re_force_line = re.compile(
    r"atom\s+(\d+)\s+type\s+\S+\s+force\s*=\s*("
    + FLOAT_RE + r")\s+(" + FLOAT_RE + r")\s+(" + FLOAT_RE + r")"
)
re_tau_line = re.compile(
    r"tau\(\s*\d+\s*\)\s*=\s*\(\s*("
    + FLOAT_RE + r")\s+(" + FLOAT_RE + r")\s+(" + FLOAT_RE + r")\s*\)"
)

# ============================================================
# HELPERS
# ============================================================

def sanitize_relpath(relpath: str) -> str:
    s = relpath.strip().replace("\\", "/")
    s = s.replace("/", "__")
    s = re.sub(r"[^A-Za-z0-9._+\-]+", "_", s)
    s = s.strip("._")
    return s or "simulation"

def choose_file(files, mode="largest"):
    if not files:
        return None
    if len(files) == 1:
        return files[0]
    if mode == "newest":
        return max(files, key=lambda p: p.stat().st_mtime)
    return max(files, key=lambda p: p.stat().st_size)

def parse_three_floats(line):
    vals = re.findall(FLOAT_RE, line)
    if len(vals) < 3:
        raise ValueError(f"Could not parse 3 floats from line:\n{line}")
    return np.array([parse_qe_float(vals[0]), parse_qe_float(vals[1]), parse_qe_float(vals[2])], dtype=np.float64)


def parse_qe_float(value):
    return float(value.replace("D", "E").replace("d", "e"))


def find_input_file_for_output(out_file):
    out_file = Path(out_file)

    same_stem = out_file.with_suffix(".in")
    if same_stem.exists():
        return same_stem

    candidates = sorted(out_file.parent.glob("*.in"))
    if len(candidates) == 0:
        return None
    if len(candidates) == 1:
        return candidates[0]

    return candidates[0]

def parse_input_cell_parameters(in_file):
    with open(in_file, "r", errors="replace") as f:
        lines = f.readlines()

    for i, line in enumerate(lines):
        low = line.strip().lower()
        if "cell_parameters" in low:
            if "angstrom" in low:
                unit = "angstrom"
            elif "bohr" in low:
                unit = "bohr"
            elif "alat" in low:
                unit = "alat"
            else:
                unit = "unknown"

            if i + 3 >= len(lines):
                raise ValueError(f"Incomplete CELL_PARAMETERS block in {in_file}")

            v1 = parse_three_floats(lines[i + 1])
            v2 = parse_three_floats(lines[i + 2])
            v3 = parse_three_floats(lines[i + 3])

            return np.vstack([v1, v2, v3]), unit

    return None, None




# ============================================================
# INITIAL STRUCTURE
# ============================================================

def parse_initial_info(lines):
    natoms = None
    alat_bohr = None
    initial_cell_alat = None
    initial_positions_alat = None
    symbols = None

    # natoms and alat
    for line in lines:
        if natoms is None:
            m = re_natoms.search(line)
            if m:
                natoms = int(m.group(1))
        if alat_bohr is None:
            m = re_alat.search(line)
            if m:
                alat_bohr = parse_qe_float(m.group(1))
        if natoms is not None and alat_bohr is not None:
            break

    # initial cell from "crystal axes: (cart. coord. in units of alat)"
    for i, line in enumerate(lines):
        if "crystal axes:" in line.lower():
            vecs = []
            for j in range(i + 1, min(i + 8, len(lines))):
                m = re_ax.search(lines[j])
                if m:
                    vecs.append([parse_qe_float(m.group(1)), parse_qe_float(m.group(2)), parse_qe_float(m.group(3))])
            if len(vecs) == 3:
                initial_cell_alat = np.asarray(vecs, dtype=np.float64)
                break

    # initial positions from "Cartesian axes" block
    for i, line in enumerate(lines):
        if line.strip() == "Cartesian axes":
            pos = []
            syms = []
            found_header = False
            for j in range(i + 1, len(lines)):
                lj = lines[j]
                if "positions (alat units)" in lj:
                    found_header = True
                    continue
                if not found_header:
                    continue

                m = re_tau_line.search(lj)
                if m:
                    parts = lj.split()
                    # example: 1 Fe tau(...) = (...)
                    syms.append(parts[1])
                    pos.append([parse_qe_float(m.group(1)), parse_qe_float(m.group(2)), parse_qe_float(m.group(3))])
                elif len(pos) > 0:
                    break

            if len(pos) > 0:
                initial_positions_alat = np.asarray(pos, dtype=np.float64)
                symbols = np.asarray(syms, dtype="U8")
                break

    if natoms is None:
        raise ValueError("Could not parse natoms")
    if alat_bohr is None:
        raise ValueError("Could not parse alat")
    if initial_cell_alat is None:
        raise ValueError("Could not parse initial cell in alat")
    if initial_positions_alat is None:
        raise ValueError("Could not parse initial positions in alat")

    return natoms, alat_bohr, initial_cell_alat, initial_positions_alat, symbols

# ============================================================
# CELL / POSITIONS / FORCES BLOCKS
# ============================================================

def parse_cell_parameters_block(lines, start_idx):
    
    header = lines[start_idx].strip().lower()
    unit = None
    if "angstrom" in header:
        unit = "angstrom"
    elif "bohr" in header:
        unit = "bohr"
    elif "alat" in header:
        unit = "alat"

    if start_idx + 3 >= len(lines):
        raise ValueError("Incomplete CELL_PARAMETERS block")

    v1 = parse_three_floats(lines[start_idx + 1])
    v2 = parse_three_floats(lines[start_idx + 2])
    v3 = parse_three_floats(lines[start_idx + 3])

    return np.vstack([v1, v2, v3]), unit, start_idx + 4

def parse_atomic_positions_block(lines, start_idx, natoms):
    header = lines[start_idx].strip().lower()
    unit = None
    if "crystal" in header:
        unit = "crystal"
    elif "angstrom" in header:
        unit = "angstrom"
    elif "bohr" in header:
        unit = "bohr"
    elif "alat" in header:
        unit = "alat"

    pos = []
    syms = []
    idx = start_idx + 1

    while idx < len(lines) and len(pos) < natoms:
        line = lines[idx].strip()
        if not line:
            idx += 1
            continue
        parts = line.split()
        if len(parts) < 4:
            break
        syms.append(parts[0])
        pos.append([parse_qe_float(parts[1]), parse_qe_float(parts[2]), parse_qe_float(parts[3])])
        idx += 1

    if len(pos) != natoms:
        raise ValueError(f"Expected {natoms} atomic positions, found {len(pos)}")

    return np.asarray(pos, dtype=np.float64), np.asarray(syms, dtype="U8"), unit, idx

def parse_forces_block(lines, start_idx, natoms):
    forces = {}
    idx = start_idx + 1

    while idx < len(lines) and len(forces) < natoms:
        line = lines[idx]
        m = re_force_line.search(line)
        if m:
            atom_index = int(m.group(1))
            if not 1 <= atom_index <= natoms:
                raise ValueError(f"Force atom index {atom_index} is outside 1..{natoms}")
            forces[atom_index] = [
                parse_qe_float(m.group(2)),
                parse_qe_float(m.group(3)),
                parse_qe_float(m.group(4)),
            ]
        elif forces and (
            "Entering Dynamics:" in line
            or "ATOMIC_POSITIONS" in line
            or "CELL_PARAMETERS" in line
            or "Total force =" in line
            or "total stress" in line.lower()
        ):
            break
        idx += 1

    missing = sorted(set(range(1, natoms + 1)) - forces.keys())
    if missing:
        preview = ", ".join(str(value) for value in missing[:12])
        suffix = "..." if len(missing) > 12 else ""
        raise ValueError(
            f"Expected forces for {natoms} atoms, found {len(forces)} unique indices; "
            f"missing [{preview}{suffix}]"
        )

    return np.asarray([forces[atom_index] for atom_index in range(1, natoms + 1)], dtype=np.float64), idx

# ============================================================
# MAIN PARSER
# ============================================================

def parse_qe_aimd_output(filepath, *, require_forces=True):
    filepath = Path(filepath)

    with open(filepath, "r", errors="replace") as f:
        lines = f.readlines()

    natoms, alat_bohr, initial_cell_alat, initial_positions_alat, symbols0 = parse_initial_info(lines)

    input_file = find_input_file_for_output(filepath)
    input_cell_parameters = None
    input_cell_unit = None

    if input_file is not None:
        input_cell_parameters, input_cell_unit = parse_input_cell_parameters(input_file)
    
    frames = []
    parse_warnings = []
    current = None
    symbols = symbols0.copy()

    i = 0
    while i < len(lines):
        line = lines[i]

        m = re_enter_dyn.search(line)
        if m:
            if current is not None and current["positions"] is not None:
                if current["forces_ry_au"] is None:
                    current["forces_ry_au"] = np.full((natoms, 3), np.nan, dtype=np.float64)
                if current["cell_parameters"] is None:
                    current["cell_parameters"] = initial_cell_alat.copy()
                    current["cell_parameters_unit"] = "alat"
                frames.append(current)

            current = {
                "iteration": int(m.group(1)),
                "time_ps": np.nan,
                "positions": None,
                "positions_unit": None,
                "forces_ry_au": None,
                "cell_parameters": None,
                "cell_parameters_unit": None,
                "energy_ry": np.nan,
                "internal_energy_ry": np.nan,
                "temperature_K": np.nan,
                "pressure_kbar": np.nan,
                "mag_total_Bohr": np.nan,
                "abs_mag_total_Bohr": np.nan,
                "ekin_ry": np.nan,
            }

            if i + 1 < len(lines):
                mt = re_time_ps.search(lines[i + 1])
                if mt:
                    current["time_ps"] = parse_qe_float(mt.group(1))

            i += 1
            continue

        if current is not None:
            if "CELL_PARAMETERS" in line:
                try:
                    cellp, unit, i = parse_cell_parameters_block(lines, i)
                    current["cell_parameters"] = cellp
                    current["cell_parameters_unit"] = unit
                    continue
                except Exception as exc:
                    parse_warnings.append(f"iteration {current['iteration']} cell near line {i + 1}: {exc}")

            if "ATOMIC_POSITIONS" in line:
                try:
                    pos, syms, unit, i = parse_atomic_positions_block(lines, i, natoms)
                    current["positions"] = pos
                    current["positions_unit"] = unit
                    symbols = syms
                    continue
                except Exception as exc:
                    parse_warnings.append(f"iteration {current['iteration']} positions near line {i + 1}: {exc}")

            if "Forces acting on atoms" in line:
                try:
                    frc, i = parse_forces_block(lines, i, natoms)
                    current["forces_ry_au"] = frc
                    continue
                except Exception as exc:
                    parse_warnings.append(f"iteration {current['iteration']} forces near line {i + 1}: {exc}")

            mt = re_temperature.search(line)
            if mt:
                current["temperature_K"] = parse_qe_float(mt.group(1))

            mk = re_ekin.search(line)
            if mk:
                current["ekin_ry"] = parse_qe_float(mk.group(1))

            me = re_total_energy.search(line)
            if me:
                current["energy_ry"] = parse_qe_float(me.group(1))

            mie = re_internal_energy.search(line)
            if mie:
                current["internal_energy_ry"] = parse_qe_float(mie.group(1))

            mp = re_pressure.search(line)
            if mp and "P=" in line:
                current["pressure_kbar"] = parse_qe_float(mp.group(1))

            mm = re_total_mag.search(line)
            if mm:
                current["mag_total_Bohr"] = parse_qe_float(mm.group(1))

            mam = re_abs_mag.search(line)
            if mam:
                current["abs_mag_total_Bohr"] = parse_qe_float(mam.group(1))

        i += 1

    if current is not None and current["positions"] is not None:
        if current["forces_ry_au"] is None:
            current["forces_ry_au"] = np.full((natoms, 3), np.nan, dtype=np.float64)
        if current["cell_parameters"] is None:
            current["cell_parameters"] = initial_cell_alat.copy()
            current["cell_parameters_unit"] = "alat"
        frames.append(current)

    if len(frames) == 0:
        return None

    positions = np.stack([fr["positions"] for fr in frames]).astype(np.float32)
    forces_ry_au = np.stack([fr["forces_ry_au"] for fr in frames]).astype(np.float32)
    cell_parameters = np.stack([fr["cell_parameters"] for fr in frames]).astype(np.float32)
    force_frame_valid = np.isfinite(forces_ry_au).all(axis=(1, 2))
    position_frame_valid = np.isfinite(positions).all(axis=(1, 2))
    frame_valid = force_frame_valid & position_frame_valid

    if require_forces and not force_frame_valid.any():
        details = parse_warnings[0] if parse_warnings else "no force blocks were found"
        raise ValueError(f"No complete force frames parsed from {filepath}: {details}")

    pos_units = [fr["positions_unit"] for fr in frames]
    cell_units = [fr["cell_parameters_unit"] for fr in frames]

    data = {
        "input_file": str(input_file) if input_file is not None else "",
        "input_cell_parameters": (
            np.asarray(input_cell_parameters, dtype=np.float32)
            if input_cell_parameters is not None
            else np.full((3, 3), np.nan, dtype=np.float32)
        ),
        "input_cell_unit": np.asarray(
            input_cell_unit if input_cell_unit is not None else "",
            dtype="U16"
        ),
        "symbols": np.asarray(symbols, dtype="U8"),
        "species": np.asarray(symbols, dtype="U8"),
        "source_file": str(filepath),
        "natoms": natoms,
        "nsteps": len(frames),
        "alat_bohr": np.float64(alat_bohr),
        "symbols": np.asarray(symbols, dtype="U8"),

        # initial structure exactly from QE header
        "initial_positions_alat": np.asarray(initial_positions_alat, dtype=np.float32),
        "initial_cell_alat": np.asarray(initial_cell_alat, dtype=np.float32),

        # MD trajectory exactly as printed by QE
        "positions": positions,                  # usually crystal for your file
        "positions_unit": np.asarray(pos_units, dtype="U16"),

        # cell parameters during MD
        "cell_parameters": cell_parameters,      # if QE printed CELL_PARAMETERS
        "cell_parameters_unit": np.asarray(cell_units, dtype="U16"),

        # observables
        "iteration": np.asarray([fr["iteration"] for fr in frames], dtype=np.int32),
        "time_ps": np.asarray([fr["time_ps"] for fr in frames], dtype=np.float32),
        "forces_ry_au": forces_ry_au,
        "force_frame_valid": force_frame_valid,
        "position_frame_valid": position_frame_valid,
        "frame_valid": frame_valid,
        "energy_ry": np.asarray([fr["energy_ry"] for fr in frames], dtype=np.float64),
        "internal_energy_ry": np.asarray([fr["internal_energy_ry"] for fr in frames], dtype=np.float64),
        "temperature_K": np.asarray([fr["temperature_K"] for fr in frames], dtype=np.float32),
        "pressure_kbar": np.asarray([fr["pressure_kbar"] for fr in frames], dtype=np.float32),
        "pressure_GPa": np.asarray([fr["pressure_kbar"] for fr in frames], dtype=np.float32) * KBAR_TO_GPA,
        "mag_total_Bohr": np.asarray([fr["mag_total_Bohr"] for fr in frames], dtype=np.float32),
        "abs_mag_total_Bohr": np.asarray([fr["abs_mag_total_Bohr"] for fr in frames], dtype=np.float32),
        "ekin_ry": np.asarray([fr["ekin_ry"] for fr in frames], dtype=np.float32),
        "parse_warnings": np.asarray(parse_warnings, dtype="U512"),
    }

    return data

# ============================================================
# SAVE
# ============================================================

def save_archive_npz(outfile, data):
    payload = {}
    metadata = {}

    for k, v in data.items():
        if isinstance(v, np.ndarray):
            payload[k] = v
        elif isinstance(v, (int, float, str)):
            metadata[k] = v
        else:
            metadata[k] = str(v)

    payload["metadata_json"] = np.array(json.dumps(metadata), dtype="U")
    np.savez_compressed(outfile, **payload)

def save_archive_pkl_xz(outfile, data):
    with lzma.open(outfile, "wb", preset=9) as f:
        pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)

# ============================================================
# WALK DIRECTORIES
# ============================================================

def find_simulation_outputs(root_dir, extensions=(".out",)):
    root_dir = Path(root_dir)
    sim_groups = []

    for dirpath, _, filenames in os.walk(root_dir):
        dirpath = Path(dirpath)
        candidates = [
            dirpath / fn
            for fn in filenames
            if fn.lower().endswith(tuple(ext.lower() for ext in extensions))
        ]
        if candidates:
            sim_groups.append((dirpath, candidates))

    return sim_groups

def process_all_simulations(root_dir, output_dir, save_fmt="npz", choose_mode="largest"):
    root_dir = Path(root_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    groups = find_simulation_outputs(root_dir, OUT_FILE_EXTENSIONS)
    manifest = []
    failed = []

    for folder, out_files in groups:
        try:
            chosen = choose_file(out_files, mode=choose_mode)
            if chosen is None:
                continue

            rel_folder = folder.relative_to(root_dir)
            base_name = sanitize_relpath(str(rel_folder))
            if len(out_files) > 1:
                base_name = f"{base_name}__{chosen.stem}"

            data = parse_qe_aimd_output(chosen)

            if data is None:
                if not SKIP_EMPTY:
                    manifest.append({
                        "folder": str(folder),
                        "source_out": str(chosen),
                        "archive": "",
                        "natoms": 0,
                        "nsteps": 0,
                        "status": "no_aimd_frames",
                    })
                continue

            if save_fmt == "npz":
                outfile = output_dir / f"{base_name}.npz"
                save_archive_npz(outfile, data)
            elif save_fmt == "pkl_xz":
                outfile = output_dir / f"{base_name}.pkl.xz"
                save_archive_pkl_xz(outfile, data)
            else:
                raise ValueError("save_fmt must be 'npz' or 'pkl_xz'")

            manifest.append({
                "folder": str(folder),
                "source_out": str(chosen),
                "archive": str(outfile),
                "natoms": int(data["natoms"]),
                "nsteps": int(data["nsteps"]),
                "valid_force_frames": int(np.count_nonzero(data["force_frame_valid"])),
                "parse_warnings": int(len(data["parse_warnings"])),
                "status": "ok",
            })

            print(f"[OK] {chosen} -> {outfile}")

        except Exception as e:
            failed.append({
                "folder": str(folder),
                "error": str(e),
            })
            print(f"[FAIL] {folder} :: {e}")

    manifest_file = output_dir / "manifest.json"
    with open(manifest_file, "w") as f:
        json.dump({"processed": manifest, "failed": failed}, f, indent=2)

    print(f"\nDone. Manifest: {manifest_file}")


# ============================================================
# RUN
# ============================================================

def parse_args():
    parser = argparse.ArgumentParser(description="Compress Quantum ESPRESSO MD outputs into validated NPZ archives.")
    parser.add_argument("root_dir", nargs="?", type=Path, default=Path(ROOT_DIR))
    parser.add_argument("--output-dir", type=Path, default=Path(OUTPUT_DIR))
    parser.add_argument("--format", choices=("npz", "pkl_xz"), default=SAVE_FMT)
    parser.add_argument("--choose", choices=("largest", "newest"), default=CHOOSE_IF_MULTIPLE)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    process_all_simulations(
        root_dir=args.root_dir,
        output_dir=args.output_dir,
        save_fmt=args.format,
        choose_mode=args.choose,
    )
