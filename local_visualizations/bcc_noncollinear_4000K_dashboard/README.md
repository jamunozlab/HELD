# Noncollinear BCC Fe AIMD Dashboard

This isolated visualization does not overwrite or modify the HELD fitting outputs in `HELD/examples`.

It reads `dataset/bcc/magnetic-non_coll/simulation.npz` and produces:

- `bcc_noncollinear_md_dashboard_final.png`: the final complete MD frame.
- `bcc_noncollinear_md_dashboard.gif`: a synchronized 64-frame dashboard covering MD iterations 3--66.

The large left panel shows moving atoms together with transparent ideal 3x3x3 BCC lattice sites. The 54 colored arrows are the evolving atom-resolved magnetization vectors integrated by QE inside atomic spheres. Arrow direction and length follow each local moment, while color represents its normalized z component. The large navy arrow is the evolved global magnetization vector reported by QE. The right panels independently show the converged global magnetization components, ionic temperature, displacement history, and the synchronized HELD phonon dispersion for the displayed frame. The orange cursor marks the displayed MD frame.

Run from the repository root with:

```bash
/opt/anaconda3/bin/python HELD/local_visualizations/bcc_noncollinear_4000K_dashboard/make_dashboard.py
```
