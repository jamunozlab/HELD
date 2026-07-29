# BCC Fe noncollinear 3x3x3 HELD diagnostic

This example processes the 54-atom noncollinear BCC trajectory in
`dataset/bcc/magnetic-non_coll/simulation.npz`.

Run from the `HELD` repository root:

```bash
python examples/iron/bcc_noncollinear_4000K_3x3x3/run_case.py
python examples/iron/bcc_noncollinear_4000K_3x3x3/make_animated_dispersion.py
```

The source output currently contains 64 complete force frames, MD iterations
3 through 66. These consecutive frames span only 0.063 ps and are strongly
correlated. The resulting mean dispersion and heatmap are early diagnostics,
not statistically converged finite-temperature phonons. A longer equilibrated
trajectory and supercell-size/cutoff tests are required for quantitative use.

The animation combines the step-resolved HELD dispersion with the atomic
configuration, ideal-to-current displacement vectors, and the converged global
magnetization vector reported by Quantum ESPRESSO.
