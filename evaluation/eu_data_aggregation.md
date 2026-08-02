# EU/AU/MSE vs. data fraction: seed-aggregated results

Context file summarizing the completed results in `results/eu_data_aggregation/*/*_test.csv`.
Each `*_test.csv` is a transposed run table produced by the EU-vs-data-fraction sweep:
rows are config fields / metrics, columns are individual runs. The filename prefix is the
`distributional_method` used (`normal`, `mixednormal`, or `mvnormal`).

* UCI tabular datasets (`concrete`, `energy`, `kin8nm`, `power`, `wine`, `yacht`) have one
  column per `(yarin_gal_uci_split_indices, data_fraction)` combination — 10 UCI seeds
  (0-9) × 5 data fractions (0.01, 0.05, 0.1, 0.25, 0.75) = 50 columns, except `yacht`
  which has no `data_fraction=0.01` runs (10 seeds × 4 fractions = 40 columns).
* The PDE datasets (`burgers` = 1D_Burgers, `ks` = 1D_KS) have a single seed per data
  fraction (5 columns, no seed averaging / no std).

**Status as of 2026-08-02:** all 8 datasets are complete. `concrete`, `energy`, `kin8nm`,
`power`, `wine`, `yacht` have both `normal` and `mixednormal` runs; `burgers` and `ks`
additionally have `mvnormal` runs.

## What's aggregated

For each dataset, method, and `data_fraction`, mean and standard deviation across seeds of:

* **MSE** — `MSETest`
* **AU** — `AleatoricUncertaintyTest`
* **EU** — `EpistemicUncertaintyTest`

All are test-set metrics (as opposed to Train/Validation columns also present in the
CSVs). For the PDE datasets (1 seed) only the point value is reported (no std).

Relative-change tables express each row as percent change from the lowest available
`data_fraction` for that dataset (`0.01` for all except `yacht`, which starts at `0.05`).

## Absolute values

### concrete — normal (n_seeds=10)

| data_fraction | MSE mean ± std | AU mean ± std | EU mean ± std |
|---|---|---|---|
| 0.01 | 174.8 ± 49.93 | 42.33 ± 21.87 | 0.004532 ± 0.003193 |
| 0.05 | 42.98 ± 15.96 | 27.48 ± 11.48 | 0.001522 ± 0.0007661 |
| 0.10 | 31.97 ± 8.453 | 22.21 ± 9.074 | 0.001536 ± 0.0004672 |
| 0.25 | 24.47 ± 5.467 | 10.65 ± 3.129 | 0.000898 ± 0.0001917 |
| 0.75 | 24.11 ± 5.155 | 8.458 ± 1.237 | 0.0009381 ± 0.0001013 |

### concrete — mixednormal (n_seeds=10)

| data_fraction | MSE mean ± std | AU mean ± std | EU mean ± std |
|---|---|---|---|
| 0.01 | 173.5 ± 64.76 | 37.77 ± 24.08 | 0.006264 ± 0.002645 |
| 0.05 | 41.29 ± 15.13 | 25.22 ± 13.12 | 0.003639 ± 0.002239 |
| 0.10 | 31.93 ± 8.548 | 19.35 ± 5.8 | 0.002273 ± 0.001415 |
| 0.25 | 24.82 ± 5.489 | 11.32 ± 4.079 | 0.001572 ± 0.0005562 |
| 0.75 | 23.82 ± 4.737 | 7.929 ± 1.515 | 0.001084 ± 0.0001467 |

### energy — normal (n_seeds=10)

| data_fraction | MSE mean ± std | AU mean ± std | EU mean ± std |
|---|---|---|---|
| 0.01 | 59.08 ± 28.71 | 17.49 ± 8.156 | 0.003036 ± 0.001606 |
| 0.05 | 2.087 ± 2.549 | 4.733 ± 1.99 | 0.0005759 ± 0.0001896 |
| 0.10 | 1.307 ± 0.91 | 2.623 ± 2.554 | 0.0004341 ± 0.0002398 |
| 0.25 | 0.4172 ± 0.1921 | 0.2635 ± 0.195 | 0.0001367 ± 5.459e-05 |
| 0.75 | 0.2413 ± 0.05675 | 0.08594 ± 0.03222 | 7.679e-05 ± 2.911e-05 |

### energy — mixednormal (n_seeds=10)

| data_fraction | MSE mean ± std | AU mean ± std | EU mean ± std |
|---|---|---|---|
| 0.01 | 90.15 ± 96.8 | 24.13 ± 22.88 | 0.008163 ± 0.005686 |
| 0.05 | 2.248 ± 3.254 | 3.819 ± 3.195 | 0.002598 ± 0.001683 |
| 0.10 | 1.306 ± 0.946 | 2.322 ± 1.831 | 0.0009522 ± 0.0004378 |
| 0.25 | 0.4033 ± 0.1866 | 0.3216 ± 0.3481 | 0.0004167 ± 0.0002567 |
| 0.75 | 0.2387 ± 0.05267 | 0.09663 ± 0.04454 | 0.0001502 ± 6.039e-05 |

### kin8nm — normal (n_seeds=10)

| data_fraction | MSE mean ± std | AU mean ± std | EU mean ± std |
|---|---|---|---|
| 0.01 | 0.008506 ± 0.001627 | 0.008111 ± 0.002437 | 0.001928 ± 0.000501 |
| 0.05 | 0.005539 ± 0.000416 | 0.003771 ± 0.0004851 | 0.001203 ± 0.0002183 |
| 0.10 | 0.005174 ± 0.0003635 | 0.003347 ± 0.0002881 | 0.001085 ± 9.6e-05 |
| 0.25 | 0.004945 ± 0.0002334 | 0.003155 ± 0.0005324 | 0.001071 ± 0.0002881 |
| 0.75 | 0.00489 ± 0.0002699 | 0.003111 ± 0.0003309 | 0.001014 ± 9.453e-05 |

### kin8nm — mixednormal (n_seeds=10)

| data_fraction | MSE mean ± std | AU mean ± std | EU mean ± std |
|---|---|---|---|
| 0.01 | 0.008266 ± 0.001438 | 0.006782 ± 0.001855 | 0.003786 ± 0.001723 |
| 0.05 | 0.005484 ± 0.0003344 | 0.004337 ± 0.001217 | 0.001672 ± 0.000428 |
| 0.10 | 0.005173 ± 0.000379 | 0.003503 ± 0.0004737 | 0.001256 ± 0.0002976 |
| 0.25 | 0.004987 ± 0.0002686 | 0.00311 ± 0.000232 | 0.00111 ± 0.000106 |
| 0.75 | 0.004914 ± 0.0002693 | 0.002989 ± 0.0002556 | 0.001018 ± 7.495e-05 |

### power — normal (n_seeds=10)

| data_fraction | MSE mean ± std | AU mean ± std | EU mean ± std |
|---|---|---|---|
| 0.01 | 17.9 ± 1.944 | 20.81 ± 5.372 | 0.001272 ± 0.0003415 |
| 0.05 | 15.99 ± 1.6 | 14.15 ± 2.193 | 0.001001 ± 0.0001201 |
| 0.10 | 15.5 ± 1.45 | 13.17 ± 1.099 | 0.001018 ± 0.0001249 |
| 0.25 | 15.43 ± 1.387 | 12.66 ± 1.035 | 0.0009793 ± 0.000128 |
| 0.75 | 15.28 ± 1.4 | 12.86 ± 0.6759 | 0.0009432 ± 8.625e-05 |

### power — mixednormal (n_seeds=10)

| data_fraction | MSE mean ± std | AU mean ± std | EU mean ± std |
|---|---|---|---|
| 0.01 | 18.58 ± 1.819 | 29.55 ± 16.21 | 0.003265 ± 0.001934 |
| 0.05 | 15.87 ± 1.593 | 15.32 ± 3.802 | 0.001404 ± 0.000312 |
| 0.10 | 15.51 ± 1.471 | 13 ± 1.579 | 0.001343 ± 0.0002747 |
| 0.25 | 15.6 ± 1.46 | 25.25 ± 38.63 | 0.001167 ± 0.0001701 |
| 0.75 | 15.32 ± 1.466 | 15.72 ± 7.041 | 0.001151 ± 8.852e-05 |

> Note: `power` / `mixednormal` at `data_fraction=0.25` has AU mean 25.25 with std 38.63
> (std > mean) — driven by at least one outlier seed; treat this cell with caution.

### wine — normal (n_seeds=10)

| data_fraction | MSE mean ± std | AU mean ± std | EU mean ± std |
|---|---|---|---|
| 0.01 | 0.7188 ± 0.2462 | 0.1842 ± 0.1097 | 0.004541 ± 0.001333 |
| 0.05 | 0.4541 ± 0.0846 | 0.228 ± 0.05499 | 0.003074 ± 0.000336 |
| 0.10 | 0.431 ± 0.07074 | 0.2456 ± 0.05024 | 0.003672 ± 0.0006212 |
| 0.25 | 0.4314 ± 0.08215 | 0.2552 ± 0.05505 | 0.003349 ± 0.0008972 |
| 0.75 | 0.4233 ± 0.07487 | 0.2462 ± 0.04643 | 0.00289 ± 0.0004568 |

### wine — mixednormal (n_seeds=10)

| data_fraction | MSE mean ± std | AU mean ± std | EU mean ± std |
|---|---|---|---|
| 0.01 | 0.7976 ± 0.3546 | 0.1625 ± 0.08073 | 0.01233 ± 0.005292 |
| 0.05 | 0.4661 ± 0.08203 | 0.244 ± 0.06452 | 0.006622 ± 0.003465 |
| 0.10 | 0.4353 ± 0.07527 | 0.2348 ± 0.04688 | 0.007685 ± 0.004871 |
| 0.25 | 0.4283 ± 0.07899 | 0.2612 ± 0.04854 | 0.004061 ± 0.000784 |
| 0.75 | 0.4258 ± 0.07612 | 0.2655 ± 0.05239 | 0.003822 ± 0.0005816 |

### yacht — normal (n_seeds=10, no data_fraction=0.01 run)

| data_fraction | MSE mean ± std | AU mean ± std | EU mean ± std |
|---|---|---|---|
| 0.05 | 57.72 ± 50.79 | 9.522 ± 11.19 | 0.001527 ± 0.001889 |
| 0.10 | 9.159 ± 5.83 | 7.748 ± 3.658 | 0.0006503 ± 0.0002385 |
| 0.25 | 4.282 ± 4.91 | 3.18 ± 2.663 | 0.000321 ± 0.0002139 |
| 0.75 | 2.053 ± 1.649 | 0.7132 ± 0.3492 | 0.0001731 ± 4.761e-05 |

### yacht — mixednormal (n_seeds=10, no data_fraction=0.01 run)

| data_fraction | MSE mean ± std | AU mean ± std | EU mean ± std |
|---|---|---|---|
| 0.05 | 36.81 ± 26.72 | 6.307 ± 4.932 | 0.001823 ± 0.0009577 |
| 0.10 | 9.226 ± 6.764 | 10.87 ± 11.55 | 0.002148 ± 0.002099 |
| 0.25 | 3.598 ± 2.744 | 3.756 ± 2.078 | 0.0009151 ± 0.0006143 |
| 0.75 | 2.08 ± 1.865 | 0.7721 ± 0.4518 | 0.0003927 ± 0.0002608 |

### burgers (1D_Burgers) — normal (n_seeds=1)

| data_fraction | MSE | AU | EU |
|---|---|---|---|
| 0.01 | 0.0001318 | 3.65e-05 | 2e-05 |
| 0.05 | 4.15e-05 | 2.14e-05 | 8e-06 |
| 0.10 | 1.7e-05 | 1.25e-05 | 4.8e-06 |
| 0.25 | 4.2e-06 | 4.6e-06 | 2.1e-06 |
| 0.75 | 7e-07 | 1.8e-06 | 1.1e-06 |

### burgers — mixednormal (n_seeds=1)

| data_fraction | MSE | AU | EU |
|---|---|---|---|
| 0.01 | 0.0001461 | 4.34e-05 | 7.34e-05 |
| 0.05 | 2.12e-05 | 1.62e-05 | 3.7e-05 |
| 0.10 | 2.26e-05 | 1.39e-05 | 1.55e-05 |
| 0.25 | 8.5e-06 | 6.1e-06 | 2.8e-06 |
| 0.75 | 6e-07 | 1.5e-06 | 1.7e-06 |

### burgers — mvnormal (n_seeds=1)

| data_fraction | MSE | AU | EU |
|---|---|---|---|
| 0.01 | 0.0001108 | 4.28e-05 | 9.93e-05 |
| 0.05 | 5.04e-05 | 2.94e-05 | 8.83e-05 |
| 0.10 | 1.15e-05 | 9.2e-06 | 7.91e-05 |
| 0.25 | 1.3e-06 | 4.4e-06 | 7.79e-05 |
| 0.75 | 4e-07 | 2.8e-06 | 7.74e-05 |

### ks (1D_KS) — normal (n_seeds=1)

| data_fraction | MSE | AU | EU |
|---|---|---|---|
| 0.01 | 0.02303 | 0.01708 | 7.3e-05 |
| 0.05 | 0.001013 | 0.002335 | 1.63e-05 |
| 0.10 | 0.000199 | 0.0007789 | 6e-06 |
| 0.25 | 9.21e-05 | 0.0003959 | 3.1e-06 |
| 0.75 | 2.25e-05 | 0.000106 | 9e-07 |

### ks — mixednormal (n_seeds=1)

| data_fraction | MSE | AU | EU |
|---|---|---|---|
| 0.01 | 0.01562 | 0.0148 | 0.0001085 |
| 0.05 | 0.005634 | 0.005568 | 6.43e-05 |
| 0.10 | 0.0002353 | 0.0008424 | 2.38e-05 |
| 0.25 | 7.45e-05 | 0.0003486 | 6.1e-06 |
| 0.75 | 1.51e-05 | 0.0001041 | 1.5e-06 |

### ks — mvnormal (n_seeds=1)

| data_fraction | MSE | AU | EU |
|---|---|---|---|
| 0.01 | 125.4 | 723.9 | 0.01574 |
| 0.05 | 21.11 | 1468 | 0.01796 |
| 0.10 | 30.9 | 1759 | 0.01784 |
| 0.25 | 28.81 | 2706 | 0.01949 |
| 0.75 | 28.71 | 2694 | 0.01955 |

> Note: `ks` / `mvnormal` MSE and AU are 3-5 orders of magnitude larger than the
> `normal`/`mixednormal` runs on the same dataset, and AU grows rather than shrinks with
> more data. This looks like a diverged/misconfigured run rather than a genuine effect —
> treat with caution and consider rerunning before using it in any writeup.

## Relative change vs. lowest data fraction

Baseline is `data_fraction=0.01` except for `yacht`, where it is `0.05` (no `0.01` runs
exist for that dataset).

### concrete — normal (baseline frac=0.01)

| data_fraction | MSE rel. | AU rel. | EU rel. |
|---|---|---|---|
| 0.01 | +0.0% | +0.0% | +0.0% |
| 0.05 | -75.4% | -35.1% | -66.4% |
| 0.10 | -81.7% | -47.5% | -66.1% |
| 0.25 | -86.0% | -74.8% | -80.2% |
| 0.75 | -86.2% | -80.0% | -79.3% |

### concrete — mixednormal (baseline frac=0.01)

| data_fraction | MSE rel. | AU rel. | EU rel. |
|---|---|---|---|
| 0.01 | +0.0% | +0.0% | +0.0% |
| 0.05 | -76.2% | -33.2% | -41.9% |
| 0.10 | -81.6% | -48.8% | -63.7% |
| 0.25 | -85.7% | -70.0% | -74.9% |
| 0.75 | -86.3% | -79.0% | -82.7% |

### energy — normal (baseline frac=0.01)

| data_fraction | MSE rel. | AU rel. | EU rel. |
|---|---|---|---|
| 0.01 | +0.0% | +0.0% | +0.0% |
| 0.05 | -96.5% | -72.9% | -81.0% |
| 0.10 | -97.8% | -85.0% | -85.7% |
| 0.25 | -99.3% | -98.5% | -95.5% |
| 0.75 | -99.6% | -99.5% | -97.5% |

### energy — mixednormal (baseline frac=0.01)

| data_fraction | MSE rel. | AU rel. | EU rel. |
|---|---|---|---|
| 0.01 | +0.0% | +0.0% | +0.0% |
| 0.05 | -97.5% | -84.2% | -68.2% |
| 0.10 | -98.6% | -90.4% | -88.3% |
| 0.25 | -99.6% | -98.7% | -94.9% |
| 0.75 | -99.7% | -99.6% | -98.2% |

### kin8nm — normal (baseline frac=0.01)

| data_fraction | MSE rel. | AU rel. | EU rel. |
|---|---|---|---|
| 0.01 | +0.0% | +0.0% | +0.0% |
| 0.05 | -34.9% | -53.5% | -37.6% |
| 0.10 | -39.2% | -58.7% | -43.7% |
| 0.25 | -41.9% | -61.1% | -44.5% |
| 0.75 | -42.5% | -61.7% | -47.4% |

### kin8nm — mixednormal (baseline frac=0.01)

| data_fraction | MSE rel. | AU rel. | EU rel. |
|---|---|---|---|
| 0.01 | +0.0% | +0.0% | +0.0% |
| 0.05 | -33.7% | -36.1% | -55.8% |
| 0.10 | -37.4% | -48.3% | -66.8% |
| 0.25 | -39.7% | -54.1% | -70.7% |
| 0.75 | -40.6% | -55.9% | -73.1% |

### power — normal (baseline frac=0.01)

| data_fraction | MSE rel. | AU rel. | EU rel. |
|---|---|---|---|
| 0.01 | +0.0% | +0.0% | +0.0% |
| 0.05 | -10.7% | -32.0% | -21.3% |
| 0.10 | -13.4% | -36.7% | -19.9% |
| 0.25 | -13.8% | -39.1% | -23.0% |
| 0.75 | -14.6% | -38.2% | -25.8% |

### power — mixednormal (baseline frac=0.01)

| data_fraction | MSE rel. | AU rel. | EU rel. |
|---|---|---|---|
| 0.01 | +0.0% | +0.0% | +0.0% |
| 0.05 | -14.6% | -48.1% | -57.0% |
| 0.10 | -16.5% | -56.0% | -58.9% |
| 0.25 | -16.0% | -14.5% | -64.3% |
| 0.75 | -17.6% | -46.8% | -64.7% |

### wine — normal (baseline frac=0.01)

| data_fraction | MSE rel. | AU rel. | EU rel. |
|---|---|---|---|
| 0.01 | +0.0% | +0.0% | +0.0% |
| 0.05 | -36.8% | +23.8% | -32.3% |
| 0.10 | -40.0% | +33.3% | -19.1% |
| 0.25 | -40.0% | +38.6% | -26.2% |
| 0.75 | -41.1% | +33.7% | -36.4% |

### wine — mixednormal (baseline frac=0.01)

| data_fraction | MSE rel. | AU rel. | EU rel. |
|---|---|---|---|
| 0.01 | +0.0% | +0.0% | +0.0% |
| 0.05 | -41.6% | +50.1% | -46.3% |
| 0.10 | -45.4% | +44.5% | -37.7% |
| 0.25 | -46.3% | +60.7% | -67.1% |
| 0.75 | -46.6% | +63.3% | -69.0% |

### yacht — normal (baseline frac=0.05)

| data_fraction | MSE rel. | AU rel. | EU rel. |
|---|---|---|---|
| 0.05 | +0.0% | +0.0% | +0.0% |
| 0.10 | -84.1% | -18.6% | -57.4% |
| 0.25 | -92.6% | -66.6% | -79.0% |
| 0.75 | -96.4% | -92.5% | -88.7% |

### yacht — mixednormal (baseline frac=0.05)

| data_fraction | MSE rel. | AU rel. | EU rel. |
|---|---|---|---|
| 0.05 | +0.0% | +0.0% | +0.0% |
| 0.10 | -74.9% | +72.3% | +17.8% |
| 0.25 | -90.2% | -40.5% | -49.8% |
| 0.75 | -94.3% | -87.8% | -78.5% |

### burgers — normal (baseline frac=0.01)

| data_fraction | MSE rel. | AU rel. | EU rel. |
|---|---|---|---|
| 0.01 | +0.0% | +0.0% | +0.0% |
| 0.05 | -68.5% | -41.4% | -60.0% |
| 0.10 | -87.1% | -65.8% | -76.0% |
| 0.25 | -96.8% | -87.4% | -89.5% |
| 0.75 | -99.5% | -95.1% | -94.5% |

### burgers — mixednormal (baseline frac=0.01)

| data_fraction | MSE rel. | AU rel. | EU rel. |
|---|---|---|---|
| 0.01 | +0.0% | +0.0% | +0.0% |
| 0.05 | -85.5% | -62.7% | -49.6% |
| 0.10 | -84.5% | -68.0% | -78.9% |
| 0.25 | -94.2% | -85.9% | -96.2% |
| 0.75 | -99.6% | -96.5% | -97.7% |

### burgers — mvnormal (baseline frac=0.01)

| data_fraction | MSE rel. | AU rel. | EU rel. |
|---|---|---|---|
| 0.01 | +0.0% | +0.0% | +0.0% |
| 0.05 | -54.5% | -31.3% | -11.1% |
| 0.10 | -89.6% | -78.5% | -20.3% |
| 0.25 | -98.8% | -89.7% | -21.6% |
| 0.75 | -99.6% | -93.5% | -22.1% |

### ks — normal (baseline frac=0.01)

| data_fraction | MSE rel. | AU rel. | EU rel. |
|---|---|---|---|
| 0.01 | +0.0% | +0.0% | +0.0% |
| 0.05 | -95.6% | -86.3% | -77.7% |
| 0.10 | -99.1% | -95.4% | -91.8% |
| 0.25 | -99.6% | -97.7% | -95.8% |
| 0.75 | -99.9% | -99.4% | -98.8% |

### ks — mixednormal (baseline frac=0.01)

| data_fraction | MSE rel. | AU rel. | EU rel. |
|---|---|---|---|
| 0.01 | +0.0% | +0.0% | +0.0% |
| 0.05 | -63.9% | -62.4% | -40.7% |
| 0.10 | -98.5% | -94.3% | -78.1% |
| 0.25 | -99.5% | -97.6% | -94.4% |
| 0.75 | -99.9% | -99.3% | -98.6% |

### ks — mvnormal (baseline frac=0.01)

| data_fraction | MSE rel. | AU rel. | EU rel. |
|---|---|---|---|
| 0.01 | +0.0% | +0.0% | +0.0% |
| 0.05 | -83.2% | +102.8% | +14.1% |
| 0.10 | -75.4% | +143.0% | +13.3% |
| 0.25 | -77.0% | +273.8% | +23.8% |
| 0.75 | -77.1% | +272.1% | +24.2% |

## Observations

* **General pattern**: on almost every dataset/method, MSE, AU, and EU all decrease
  monotonically (or near-monotonically) as `data_fraction` increases, consistent with
  more training data improving both accuracy and calibrated uncertainty. The PDE
  datasets (`burgers`, `ks`, `normal`/`mixednormal`) show the steepest relative drops
  (often >95% by `data_fraction=0.75`), while `power` is the flattest responder
  (MSE/EU only move ~15-25% across the whole range).
* **wine is the outlier for AU**: under both `normal` and `mixednormal`, AU *increases*
  by 24-63% relative to the `0.01` baseline instead of decreasing, even as MSE and EU
  drop normally. This was already visible in the two-dataset snapshot and persists now
  with the full data — worth investigating (e.g. whether the aleatoric head is
  underfitting at low `data_fraction` for this dataset specifically).
* **yacht mixednormal** shows a similar non-monotonic AU/EU bump at `data_fraction=0.10`
  (AU +72%, EU +18% vs. the `0.05` baseline) before dropping sharply — likely a
  high-variance small-dataset effect (`yacht` is the smallest UCI dataset here).
* **Data-quality caveats**: `power`/`mixednormal` at `data_fraction=0.25` has an AU std
  larger than its mean (outlier seed), and `ks`/`mvnormal` produces MSE/AU 3-5 orders of
  magnitude larger than the `normal`/`mixednormal` runs with AU *increasing* with more
  data — this run looks diverged/misconfigured rather than a genuine result and should
  be rerun or excluded before drawing conclusions from `mvnormal` on `ks`.

## Reproduction

Computed directly from the transposed `*_test.csv` files with pandas: load with
`index_col=0`, then group columns by the `data_fraction` row (equivalently, by
`yarin_gal_uci_split_indices` for the UCI datasets) and average the `MSETest` /
`AleatoricUncertaintyTest` / `EpistemicUncertaintyTest` rows within each group. Relative
change is `(value - baseline) / baseline * 100` against the lowest `data_fraction` group
for that dataset.
