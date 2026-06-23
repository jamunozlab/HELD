# HELD

HELD workflow for monatomic `bcc`, `fcc`, and `hcp` datasets stored as NPZ trajectories (at the moment, It will be able to read md-based simulations from qe and lammps in near future).

The main entrypoint is `HELD.py`. It supports:

- `fit`: fit HELD coefficients from an NPZ trajectory
- `dispersion`: compute and plot the mean HELD dispersion along a built-in high-symmetry path
- `heatmap`: compute and plot the HELD dispersi on heat map from all fitted frames


Notes:

- The current unified implementation is phase-generic and element-generic for single-element datasets such as Fe, Ni, Co, or Al.
- The code infers the primitive cell and supercell repetition directly from the NPZ geometry, so the workflow is not tied to hardcoded Fe paths.
- Example iron runs and plots are stored under `examples/iron/`.
- The notebook `notebooks/iron_three_phase_examples.ipynb` shows one `bcc`, one `fcc`, and one `hcp` Fe case using `skip=100` and `aggregate="mean"`.

Minimal usage:

```bash
/opt/anaconda3/bin/python HELD.py fit \
  --phase bcc \
  --npz /path/to/case.npz \
  --output-csv results/held_case.csv \
  --aggregate mean \
  --skip 100
```

```bash
/opt/anaconda3/bin/python HELD.py dispersion \
  --phase bcc \
  --npz /path/to/case.npz \
  --held-csv results/held_case.csv \
  --output-data results/held_case_dispersion.dat \
  --output-plot results/held_case_dispersion.png
```

```bash
/opt/anaconda3/bin/python HELD.py heatmap \
  --phase bcc \
  --npz /path/to/case.npz \
  --held-csv results/held_case.csv \
  --cache-npz results/held_case_heatmap_steps.npz \
  --output-plot results/held_case_heatmap.png
```
