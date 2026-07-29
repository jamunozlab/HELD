# Collinear-magnetic BCC Fe at 4000 K

This example applies HELD to the updated spin-polarized BCC Fe trajectory:

- source trajectory: `../../../../dataset/bcc/magnetic-collinear/simulation.npz`
- ideal BCC reference: `../../../../dataset/bcc/magnetic-collinear/simulation-ideal.npz`
- supercell: `4 x 4 x 4` conventional BCC cells
- atoms: `128`
- valid fitted frames: `112`
- aggregate: arithmetic mean of the per-frame HELD coefficients
- interaction shells: `5`

Run the complete workflow from the HELD repository root:

```bash
/opt/anaconda3/bin/python examples/iron/bcc_collinear_4000K/run_case.py
```

Generate the step-resolved animated dispersion:

```bash
/opt/anaconda3/bin/python examples/iron/bcc_collinear_4000K/make_animated_dispersion.py
```

The runner creates `simulation_held_ideal_reference.npz` by combining the full updated trajectory with the ideal BCC reference positions. It also converts the artificial per-atom QE labels to the physical symbol `Fe`.

## Step-resolved observables

`held_steps_with_thermodynamics.csv` contains one row per fitted frame with:

- source frame index and MD iteration,
- time,
- temperature,
- pressure,
- total magnetization,
- absolute magnetization,
- and the 13 fitted HELD coefficients.

The CSV was checked row by row against the source NPZ. All `112` fitted rows preserve the original trajectory values.

| Observable | Mean | Minimum | Maximum |
|---|---:|---:|---:|
| Temperature (K) | `4038.63` | `3510.24` | `4626.28` |
| Pressure (GPa) | `125.884` | `123.108` | `128.463` |
| Total magnetization (Bohr magnetons/cell) | `10.815` | `0.720` | `16.250` |
| Absolute magnetization (Bohr magnetons/cell) | `27.318` | `21.060` | `34.580` |

## HELD result

The mean HELD dispersion spans `-0.178` to `14.106 THz`. The small negative dip is localized near the acoustic branch close to Gamma. It is much smaller than the overall phonon bandwidth, but it should be retained and reported rather than silently clipped. Its sensitivity to trajectory length, frame selection, shell count, and coefficient aggregation should be tested before assigning physical significance.

The heatmap stores `112 x 301 x 3` step-, q-point-, and branch-resolved frequencies.

## Outputs

- `held_mean_all_valid.csv`: coefficient-only HELD file used by the dispersion and heatmap readers
- `held_steps_with_thermodynamics.csv`: per-step observables and HELD coefficients
- `held_dispersion_all_valid_mean.dat`: mean dispersion data
- `held_dispersion_all_valid_mean.png`: mean dispersion plot
- `held_heatmap_steps_all_valid.npz`: per-step frequency cache
- `held_heatmap_all_valid_mean.png`: finite-temperature HELD heatmap
- `held_temperature_pressure_magnetization_by_step.png`: trajectory diagnostic
- `bcc_collinear_4000K_HELD_0001_0112.gif`: animated per-step dispersion with temperature, pressure, and magnetization
- `held_run_summary.json`: machine-readable settings, statistics, and output paths
