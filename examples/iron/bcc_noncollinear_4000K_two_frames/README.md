# Noncollinear BCC Fe at 4000 K: two-frame HELD diagnostic

This example fits HELD force constants to the two complete noncollinear BCC Fe
force snapshots in:

`../../../../dataset/bcc/magnetic-non_coll/simulation.npz`

The ideal `4 x 4 x 4` BCC reference from `simulation-ideal.npz` is used to
recover the correct lattice symmetry. Both 128-atom configurations are retained.

Run from the HELD repository root:

```bash
/opt/anaconda3/bin/python examples/iron/bcc_noncollinear_4000K_two_frames/run_case.py
```

## Outputs

- `held_mean_two_frames.csv`: mean and snapshot-resolved HELD coefficients
- `held_steps_with_thermodynamics.csv`: coefficients with temperature, pressure, and magnetization
- `held_dispersion_two_frame_mean.dat`: mean phonon dispersion
- `held_dispersion_two_frame_mean.png`: mean phonon dispersion plot
- `held_dispersion_two_snapshots.npz`: both snapshot dispersions and their mean
- `held_dispersion_two_snapshots_and_mean.png`: direct comparison of both snapshots and their mean
- `held_run_summary.json`: settings, frequency ranges, snapshot spread, and limitations

## Limitation

The two snapshots provide many force equations but only two statistically
independent configurations. This is a diagnostic fit, not a converged
finite-temperature phonon calculation. Additional decorrelated AIMD frames are
required before using the mean dispersion quantitatively.
