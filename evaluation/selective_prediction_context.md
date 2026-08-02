# Uncertainty quantification for diffusion models: selective prediction results

Context file for writing about selective prediction with a distributional diffusion
model. Derived from the per-example arrays in `results/selective_prediction/*.npz`,
produced by `evaluation/selective_prediction.py` and analysed in
`evaluation/selective_prediction.ipynb`.

---

## 1. What the experiment does

**Question.** If a model can say how uncertain it is, can that uncertainty be used to
decide which predictions to keep and which to hand off? A useful uncertainty ranks the
test set so that rejecting the most uncertain examples removes the largest errors first.

**Setup.** Every test example is a **single one-step prediction** (no autoregressive
rollout). For each example the model draws 100 diffusion samples; from those we compute

* `mse` — squared error of the mean prediction against the target, averaged over the
  whole spatial domain (and channels);
* `au` — aleatoric uncertainty, a variance;
* `eu` — epistemic uncertainty, a variance;
* `tu = au + eu` — total uncertainty.

All four are variances on the same scale, so the decomposition is additive and the
methods are directly comparable.

**Selective prediction / retention curves.** Examples are sorted by an uncertainty
measure ascending. At retention rate *r* the `1 - r` most uncertain examples are
rejected and the mean MSE over the retained ones is recorded, for *r* on a 51-point grid
over `[0.5, 1]`. Two baselines bracket the result:

* **Oracle** — rank by the realised error itself (the best any ranking can do);
* **Random** — rejection independent of error, so the curve is flat at the overall
  mean MSE.

**PRR (prediction–rejection ratio).** The area between the method's retention curve and
the random baseline, normalised by the area between oracle and random, over
`r ∈ [0.5, 1]`:

```
PRR = (A_random - A_method) / (A_random - A_oracle)
```

* `PRR = 1` — the uncertainty ranks examples as well as the realised error itself;
* `PRR = 0` — no better than rejecting at random;
* `PRR < 0` — worse than random (the uncertainty is anti-correlated with the error).

**ρ (rank correlation with the error).** Spearman correlation between the uncertainty
measure and the per-example squared error, computed directly from the saved arrays.
PRR and ρ measure the same thing from different angles: PRR only integrates over
retention ≥ 0.5 (it weights the top of the ranking), ρ weights the whole ranking
uniformly.

---

## 2. The four methods being compared

All four are trained on the same data and share the same distributional-diffusion
scaffold except the ensemble, which is the deterministic diffusion aggregated over
independently trained checkpoints.

| | **Diagonal** (`normal`) | **Mixture** (`mixednormal`) | **Multivariate** (`mvnormal`) | **Ensemble** |
|---|---|---|---|---|
| Head | Normal: the network predicts a mean and a standard deviation for the noise at every reverse step | Normal-mixture: a weighted mixture of Normals per step | Multivariate Normal (low-rank / LoRA covariance over the domain) per step | none — deterministic diffusion, 5 independently trained checkpoints (seeds 1–5) |
| Aleatoric uncertainty | variance across the 100 drawn samples | variance across the 100 drawn samples | variance across the 100 drawn samples | mean over members of the within-member sample variance |
| Epistemic uncertainty | the model's predicted noise variance, weighted by the reverse-step coefficient `(1-α)²/(α(1-α̂))` and averaged over the trajectory | the mixture's *marginal* variance (weighted mean of component variances plus spread of component means) — the same per-point quantity as Diagonal, same weighting/averaging | only the **diagonal** of the predicted noise covariance (off-diagonal correlations are dropped); same weighting/averaging as Diagonal | disagreement between the 5 checkpoint means, i.e. variance across checkpoints |
| Trained checkpoints per dataset | 1 | 1 (`n_components`: 50 for KS, 2 for Burgers, 10 for T2M) | 1, LoRA rank 1 — **not trained for UCI** (scalar targets would make it coincide with Diagonal) | 5 — **not trained for T2M** (only one deterministic checkpoint ever existed) |
| Cost at inference | 1 model, 1 sampling pass | 1 model, 1 sampling pass | 1 model, 1 sampling pass | 5 models |

The key methodological point is unchanged from the two-method version of this
experiment: every distributional head gets an epistemic signal from a **single
training run**, where the ensemble needs five. This experiment asks whether that
single-model signal — under three different parameterisations of the predictive
distribution — is as usable for selective prediction as the ensemble's.

Both models use the law of total variance to split their uncertainty, all four report
`tu = au + eu`, and all four get the same total sample budget (100 samples: the
ensemble draws 20 per member × 5 members).

**Availability.** Not every method has been evaluated on every dataset yet:

* **T2M** — only Diagonal has selective-prediction results. Mixture and Multivariate
  checkpoints exist but have not been run through `selective_prediction.py`; there is
  no ensemble at all (see above).
* **UCI (all 8 datasets)** — Diagonal and Mixture only. Multivariate was never trained
  there; Ensemble uses 5 seed checkpoints on Yarin-Gal split 0.
* **KS / Burgers** — all four methods have results.

---

## 3. Datasets

Eleven test sets in three families:

**PDE fields (1D)** — `KS` (Kuramoto–Sivashinsky), `Burgers`. UNet backbone, 1000 test
examples each, one random timestep drawn per example (frozen under a fixed seed so all
methods see identical data).

**Weather field (2D)** — `T2M` (WeatherBench 2-metre temperature, 160×220 grid), 729 test
examples, UNet backbone. Distributional (Diagonal) only, per the availability note
above.

**UCI regression** — `concrete`, `energy`, `kin8nm`, `naval`, `power`, `protein`, `wine`,
`yacht`. MLP/CARD backbone, evaluated on Yarin-Gal **split 0**. All five ensemble members
were trained on split 0 with seeds 1–5, so they differ only in initialisation and the
split-0 test set is held out from every member — the same held-out data the
distributional heads are evaluated on, with the same split-0 CARD regressor.

Test-set sizes range from 31 (`yacht`) to 4573 (`protein`); the small ones carry
correspondingly noisy estimates.

**Units.** MSE is in standardised units for every dataset (targets are normalised), which
is why the field datasets sit at `1e-6` and the UCI ones at `1e-2`–`1e-1`. MSE
magnitudes are **not comparable across datasets**; PRR and ρ are.

---

## 4. Main results table

`n` = test examples. `MSE` = mean over the test set. `PRR` = prediction–rejection ratio
(1 = oracle, 0 = random). `ρ` = Spearman rank correlation between that uncertainty and
the per-example squared error.

| Dataset | Method | n | MSE | PRR Total | PRR Aleatoric | PRR Epistemic | ρ Total | ρ Aleatoric | ρ Epistemic |
|---|---|---|---|---|---|---|---|---|---|
| KS | Diagonal | 1000 | 2.22e-06 | 0.8402 | 0.8394 | 0.8453 | 0.8105 | 0.8102 | 0.8139 |
| KS | Mixture | 1000 | 1.67e-06 | 0.8463 | 0.8458 | 0.8135 | 0.7706 | 0.7718 | 0.7284 |
| KS | Multivariate | 1000 | 3.36e-06 | 0.2089 | 0.2081 | 0.4088 | 0.1309 | 0.1306 | 0.4105 |
| KS | Ensemble | 1000 | 2.47e-06 | 0.7937 | 0.6881 | 0.8177 | 0.7678 | 0.6734 | 0.7764 |
| Burgers | Diagonal | 1000 | 3.84e-06 | 0.9945 | 0.9945 | 0.9945 | 0.9181 | 0.9177 | 0.9136 |
| Burgers | Mixture | 1000 | 6.27e-06 | 0.9927 | 0.9927 | 0.9922 | 0.9272 | 0.9265 | 0.9344 |
| Burgers | Multivariate | 1000 | 3.84e-06 | 0.9456 | 0.9453 | 0.9708 | 0.2889 | 0.2880 | 0.5354 |
| Burgers | Ensemble | 1000 | 4.63e-06 | 0.9921 | 0.9899 | 0.9942 | 0.8570 | 0.7774 | 0.8860 |
| T2M | Diagonal | 729 | 6.12e-03 | 0.7286 | 0.7286 | 0.6951 | 0.7119 | 0.7123 | 0.6418 |
| UCI_concrete | Diagonal | 103 | 7.26e-02 | 0.3953 | 0.3903 | 0.3803 | 0.3108 | 0.3122 | 0.3066 |
| UCI_concrete | Mixture | 103 | 7.09e-02 | 0.2411 | 0.2426 | 0.2622 | 0.3324 | 0.3357 | 0.2992 |
| UCI_concrete | Ensemble | 103 | 7.34e-02 | 0.2198 | 0.2197 | 0.0939 | 0.2908 | 0.3175 | -0.0080 |
| UCI_energy | Diagonal | 77 | 1.22e-03 | 0.4986 | 0.5015 | 0.4161 | 0.4708 | 0.4727 | 0.4472 |
| UCI_energy | Mixture | 77 | 1.23e-03 | 0.3797 | 0.3769 | 0.4263 | 0.4599 | 0.4612 | 0.4802 |
| UCI_energy | Ensemble | 77 | 1.17e-03 | 0.0820 | 0.2224 | 0.0420 | 0.2639 | 0.3660 | 0.1979 |
| UCI_kin8nm | Diagonal | 819 | 6.92e-02 | 0.2083 | 0.2072 | 0.2500 | 0.1461 | 0.1454 | 0.2069 |
| UCI_kin8nm | Mixture | 819 | 6.86e-02 | 0.2510 | 0.2508 | 0.2352 | 0.1783 | 0.1778 | 0.1895 |
| UCI_kin8nm | Ensemble | 819 | 6.91e-02 | 0.2526 | 0.2629 | 0.0259 | 0.1986 | 0.2046 | 0.0396 |
| UCI_naval | Diagonal | 1193 | 6.03e-05 | 0.7895 | 0.7877 | 0.7709 | 0.4006 | 0.4013 | 0.3238 |
| UCI_naval | Mixture | 1193 | 5.94e-05 | 0.7705 | 0.7703 | 0.7790 | 0.3091 | 0.3099 | 0.2852 |
| UCI_naval | Ensemble | 1193 | 6.58e-05 | 0.5707 | 0.7382 | 0.1661 | 0.2131 | 0.3084 | 0.1020 |
| UCI_power | Diagonal | 957 | 5.01e-02 | 0.2213 | 0.2210 | 0.2066 | 0.3156 | 0.3149 | 0.3206 |
| UCI_power | Mixture | 957 | 4.61e-02 | 0.1488 | 0.1486 | 0.1015 | 0.3225 | 0.3227 | 0.2890 |
| UCI_power | Ensemble | 957 | 5.29e-02 | 0.1661 | 0.1593 | 0.0628 | 0.3371 | 0.3261 | 0.1981 |
| UCI_protein | Diagonal | 4573 | 3.60e-01 | 0.4283 | 0.4284 | 0.3517 | 0.6751 | 0.6748 | 0.6594 |
| UCI_protein | Mixture | 4573 | 3.60e-01 | 0.3944 | 0.3943 | 0.3734 | 0.6429 | 0.6424 | 0.6638 |
| UCI_protein | Ensemble | 4573 | 3.64e-01 | 0.4396 | 0.4363 | 0.3428 | 0.6495 | 0.6481 | 0.5364 |
| UCI_wine | Diagonal | 160 | 6.17e-01 | 0.0273 | 0.0250 | 0.1002 | 0.4301 | 0.4283 | 0.4003 |
| UCI_wine | Mixture | 160 | 4.99e-01 | 0.0582 | 0.0559 | 0.1343 | 0.4298 | 0.4313 | 0.3837 |
| UCI_wine | Ensemble | 160 | 5.06e-01 | 0.0953 | -0.0117 | 0.1794 | 0.5726 | 0.4794 | 0.5628 |
| UCI_yacht | Diagonal | 31 | 2.45e-03 | 0.9334 | 0.9296 | 0.9527 | 0.7024 | 0.6956 | 0.6383 |
| UCI_yacht | Mixture | 31 | 2.15e-03 | 0.8540 | 0.8449 | 0.8658 | 0.4169 | 0.3980 | 0.4468 |
| UCI_yacht | Ensemble | 31 | 2.74e-03 | 0.9424 | 0.9707 | 0.7722 | 0.6649 | 0.5222 | 0.6048 |

### 4b. PRR side by side, methods as columns

**PRR Epistemic**

| Dataset | Diagonal | Mixture | Multivariate | Ensemble |
|---|---|---|---|---|
| KS | 0.8453 | 0.8135 | 0.4088 | 0.8177 |
| Burgers | 0.9945 | 0.9922 | 0.9708 | 0.9942 |
| T2M | 0.6951 | — | — | — |
| UCI_concrete | 0.3803 | 0.2622 | — | 0.0939 |
| UCI_energy | 0.4161 | 0.4263 | — | 0.0420 |
| UCI_kin8nm | 0.2500 | 0.2352 | — | 0.0259 |
| UCI_naval | 0.7709 | 0.7790 | — | 0.1661 |
| UCI_power | 0.2066 | 0.1015 | — | 0.0628 |
| UCI_protein | 0.3517 | 0.3734 | — | 0.3428 |
| UCI_wine | 0.1002 | 0.1343 | — | 0.1794 |
| UCI_yacht | 0.9527 | 0.8658 | — | 0.7722 |

**PRR Aleatoric**

| Dataset | Diagonal | Mixture | Multivariate | Ensemble |
|---|---|---|---|---|
| KS | 0.8394 | 0.8458 | 0.2081 | 0.6881 |
| Burgers | 0.9945 | 0.9927 | 0.9453 | 0.9899 |
| T2M | 0.7286 | — | — | — |
| UCI_concrete | 0.3903 | 0.2426 | — | 0.2197 |
| UCI_energy | 0.5015 | 0.3769 | — | 0.2224 |
| UCI_kin8nm | 0.2072 | 0.2508 | — | 0.2629 |
| UCI_naval | 0.7877 | 0.7703 | — | 0.7382 |
| UCI_power | 0.2210 | 0.1486 | — | 0.1593 |
| UCI_protein | 0.4284 | 0.3943 | — | 0.4363 |
| UCI_wine | 0.0250 | 0.0559 | — | -0.0117 |
| UCI_yacht | 0.9296 | 0.8449 | — | 0.9707 |

**PRR Total**

| Dataset | Diagonal | Mixture | Multivariate | Ensemble |
|---|---|---|---|---|
| KS | 0.8402 | 0.8463 | 0.2089 | 0.7937 |
| Burgers | 0.9945 | 0.9927 | 0.9456 | 0.9921 |
| T2M | 0.7286 | — | — | — |
| UCI_concrete | 0.3953 | 0.2411 | — | 0.2198 |
| UCI_energy | 0.4986 | 0.3797 | — | 0.0820 |
| UCI_kin8nm | 0.2083 | 0.2510 | — | 0.2526 |
| UCI_naval | 0.7895 | 0.7705 | — | 0.5707 |
| UCI_power | 0.2213 | 0.1488 | — | 0.1661 |
| UCI_protein | 0.4283 | 0.3944 | — | 0.4396 |
| UCI_wine | 0.0273 | 0.0582 | — | 0.0953 |
| UCI_yacht | 0.9334 | 0.8540 | — | 0.9424 |

**MSE (standardised units, not comparable across datasets)**

| Dataset | Diagonal | Mixture | Multivariate | Ensemble |
|---|---|---|---|---|
| KS | 2.22e-06 | 1.67e-06 | 3.36e-06 | 2.47e-06 |
| Burgers | 3.84e-06 | 6.27e-06 | 3.84e-06 | 4.63e-06 |
| T2M | 6.12e-03 | — | — | — |
| UCI_concrete | 7.26e-02 | 7.09e-02 | — | 7.34e-02 |
| UCI_energy | 1.22e-03 | 1.23e-03 | — | 1.17e-03 |
| UCI_kin8nm | 6.92e-02 | 6.86e-02 | — | 6.91e-02 |
| UCI_naval | 6.03e-05 | 5.94e-05 | — | 6.58e-05 |
| UCI_power | 5.01e-02 | 4.61e-02 | — | 5.29e-02 |
| UCI_protein | 3.60e-01 | 3.60e-01 | — | 3.64e-01 |
| UCI_wine | 6.17e-01 | 4.99e-01 | — | 5.06e-01 |
| UCI_yacht | 2.45e-03 | 2.15e-03 | — | 2.74e-03 |

---

## 5. Agreement with the ensemble

Do the distributional heads order the *same* test examples the same way as the
ensemble? For each dataset and each distributional head, its AU/EU/TU is correlated
against the ensemble's, per test example. Both runs used the same seed and sample
count, and a `target_mean` fingerprint saved with each run confirms they evaluated
identical examples in identical order.

* **Pearson** — linear agreement of raw magnitudes. All are variances, but a
  distributional head's is a predicted noise variance propagated through the reverse
  trajectory while the ensemble's is a variance between checkpoints; there is no reason
  for the constant relating them to be 1, and Pearson penalises that.
* **Spearman** — rank agreement. **This is the one that matters for selective
  prediction**: retention curves only use the *ordering*, so two measures with Spearman
  1 would reject exactly the same examples however differently they are scaled.

T2M is absent (no ensemble). Multivariate is absent for UCI (never trained there).

| Dataset | Method | n | AU Pearson | AU Spearman | EU Pearson | EU Spearman | TU Pearson | TU Spearman |
|---|---|---|---|---|---|---|---|---|
| KS | Diagonal | 1000 | 0.933 | 0.964 | 0.834 | 0.821 | 0.947 | 0.921 |
| KS | Mixture | 1000 | 0.890 | 0.930 | 0.742 | 0.753 | 0.895 | 0.884 |
| KS | Multivariate | 1000 | 0.277 | 0.179 | 0.455 | 0.523 | 0.274 | 0.191 |
| Burgers | Diagonal | 1000 | 0.843 | 0.818 | 0.648 | 0.890 | 0.790 | 0.884 |
| Burgers | Mixture | 1000 | 0.846 | 0.838 | 0.647 | 0.889 | 0.794 | 0.884 |
| Burgers | Multivariate | 1000 | 0.915 | 0.278 | 0.633 | 0.771 | 0.859 | 0.281 |
| UCI_concrete | Diagonal | 103 | 0.755 | 0.810 | 0.160 | 0.151 | 0.707 | 0.760 |
| UCI_concrete | Mixture | 103 | 0.876 | 0.855 | 0.224 | 0.177 | 0.838 | 0.802 |
| UCI_energy | Diagonal | 77 | 0.489 | 0.766 | 0.605 | 0.572 | 0.610 | 0.667 |
| UCI_energy | Mixture | 77 | 0.431 | 0.780 | 0.525 | 0.516 | 0.518 | 0.652 |
| UCI_kin8nm | Diagonal | 819 | 0.825 | 0.828 | 0.413 | 0.390 | 0.824 | 0.831 |
| UCI_kin8nm | Mixture | 819 | 0.849 | 0.853 | 0.424 | 0.403 | 0.854 | 0.858 |
| UCI_naval | Diagonal | 1193 | 0.794 | 0.680 | 0.183 | 0.567 | 0.472 | 0.702 |
| UCI_naval | Mixture | 1193 | 0.783 | 0.709 | 0.126 | 0.490 | 0.421 | 0.574 |
| UCI_power | Diagonal | 957 | 0.446 | 0.562 | 0.123 | 0.160 | 0.432 | 0.536 |
| UCI_power | Mixture | 957 | 0.446 | 0.525 | 0.099 | 0.181 | 0.430 | 0.501 |
| UCI_protein | Diagonal | 4573 | 0.824 | 0.854 | 0.413 | 0.663 | 0.820 | 0.854 |
| UCI_protein | Mixture | 4573 | 0.703 | 0.768 | 0.336 | 0.621 | 0.695 | 0.765 |
| UCI_wine | Diagonal | 160 | 0.428 | 0.455 | 0.084 | 0.365 | 0.410 | 0.412 |
| UCI_wine | Mixture | 160 | 0.326 | 0.394 | 0.178 | 0.337 | 0.354 | 0.429 |
| UCI_yacht | Diagonal | 31 | 0.646 | 0.700 | 0.724 | 0.725 | 0.820 | 0.796 |
| UCI_yacht | Mixture | 31 | 0.471 | 0.619 | 0.684 | 0.486 | 0.659 | 0.792 |

---

## 6. Uncertainty magnitudes

Useful for explaining *why* Total and Aleatoric are nearly identical for Diagonal and
Mixture: their EU is a small fraction of their TU, so `tu ≈ au`. Multivariate is the
exception — see Section 7.

| Dataset | Method | mean MSE | mean AU | mean EU | mean TU | EU share of TU |
|---|---|---|---|---|---|---|
| KS | Diagonal | 2.22e-06 | 1.36e-05 | 5.21e-07 | 1.41e-05 | 0.037 |
| KS | Mixture | 1.67e-06 | 1.36e-05 | 1.51e-06 | 1.51e-05 | 0.100 |
| KS | Multivariate | 3.36e-06 | 3.82e-05 | 2.45e-04 | 2.83e-04 | 0.865 |
| KS | Ensemble | 2.47e-06 | 2.35e-06 | 2.51e-06 | 4.86e-06 | 0.517 |
| Burgers | Diagonal | 3.84e-06 | 9.79e-06 | 3.60e-07 | 1.02e-05 | 0.035 |
| Burgers | Mixture | 6.27e-06 | 1.58e-05 | 5.53e-07 | 1.64e-05 | 0.034 |
| Burgers | Multivariate | 3.84e-06 | 2.94e-05 | 7.81e-05 | 1.08e-04 | 0.727 |
| Burgers | Ensemble | 4.63e-06 | 6.58e-06 | 4.14e-06 | 1.07e-05 | 0.386 |
| T2M | Diagonal | 6.12e-03 | 3.00e-03 | 5.61e-05 | 3.05e-03 | 0.018 |
| UCI_concrete | Diagonal | 7.26e-02 | 1.41e-02 | 3.95e-04 | 1.45e-02 | 0.027 |
| UCI_concrete | Mixture | 7.09e-02 | 1.31e-02 | 4.57e-04 | 1.35e-02 | 0.034 |
| UCI_concrete | Ensemble | 7.34e-02 | 8.09e-03 | 1.74e-03 | 9.82e-03 | 0.177 |
| UCI_energy | Diagonal | 1.22e-03 | 4.24e-04 | 4.04e-05 | 4.65e-04 | 0.087 |
| UCI_energy | Mixture | 1.23e-03 | 4.46e-04 | 3.58e-05 | 4.82e-04 | 0.074 |
| UCI_energy | Ensemble | 1.17e-03 | 7.60e-05 | 2.17e-04 | 2.93e-04 | 0.741 |
| UCI_kin8nm | Diagonal | 6.92e-02 | 3.84e-02 | 8.59e-04 | 3.92e-02 | 0.022 |
| UCI_kin8nm | Mixture | 6.86e-02 | 3.97e-02 | 8.85e-04 | 4.06e-02 | 0.022 |
| UCI_kin8nm | Ensemble | 6.91e-02 | 2.77e-02 | 1.94e-03 | 2.97e-02 | 0.065 |
| UCI_naval | Diagonal | 6.03e-05 | 5.45e-05 | 4.12e-06 | 5.86e-05 | 0.070 |
| UCI_naval | Mixture | 5.94e-05 | 7.76e-05 | 5.41e-06 | 8.30e-05 | 0.065 |
| UCI_naval | Ensemble | 6.58e-05 | 3.26e-06 | 1.43e-05 | 1.76e-05 | 0.815 |
| UCI_power | Diagonal | 5.01e-02 | 3.11e-02 | 6.26e-04 | 3.17e-02 | 0.020 |
| UCI_power | Mixture | 4.61e-02 | 2.84e-02 | 6.16e-04 | 2.90e-02 | 0.021 |
| UCI_power | Ensemble | 5.29e-02 | 2.93e-02 | 3.44e-03 | 3.28e-02 | 0.105 |
| UCI_protein | Diagonal | 3.60e-01 | 2.58e-01 | 1.68e-03 | 2.60e-01 | 0.006 |
| UCI_protein | Mixture | 3.60e-01 | 2.61e-01 | 1.73e-03 | 2.63e-01 | 0.007 |
| UCI_protein | Ensemble | 3.64e-01 | 2.99e-01 | 2.54e-02 | 3.25e-01 | 0.078 |
| UCI_wine | Diagonal | 6.17e-01 | 1.40e-01 | 5.19e-04 | 1.41e-01 | 0.004 |
| UCI_wine | Mixture | 4.99e-01 | 1.04e-01 | 6.03e-04 | 1.04e-01 | 0.006 |
| UCI_wine | Ensemble | 5.06e-01 | 1.62e-01 | 8.01e-02 | 2.42e-01 | 0.331 |
| UCI_yacht | Diagonal | 2.45e-03 | 1.96e-04 | 2.00e-05 | 2.16e-04 | 0.093 |
| UCI_yacht | Mixture | 2.15e-03 | 1.62e-04 | 2.01e-05 | 1.82e-04 | 0.110 |
| UCI_yacht | Ensemble | 2.74e-03 | 4.04e-05 | 1.08e-04 | 1.48e-04 | 0.728 |

---

## 7. What the numbers show

Ten datasets admit a Diagonal-vs-Mixture-vs-Ensemble comparison (all but T2M);
Multivariate is only comparable on KS and Burgers.

**Mixture usually wins on point accuracy, not on ranking.** Mixture has the lower MSE
on **8 of the 10** non-T2M datasets (`concrete`, `kin8nm`, `naval`, `power`, `protein`,
`wine`, `yacht`, and `KS`), sometimes clearly (`wine`: 0.499 vs Diagonal's 0.617).
Diagonal wins only on `Burgers`, Ensemble only on `energy`. But better point predictions
do not translate into a better *uncertainty ranking*: on PRR Epistemic, **Diagonal wins
6 of 10** (`KS`, `Burgers`, `concrete`, `kin8nm`, `power`, `yacht`), Mixture wins 3
(`energy`, `naval`, `protein`, each narrowly), and Ensemble wins 1 (`wine`). The two
heads are close on most datasets and the ranking is not simply "whichever head fits
better wins" — `wine` is won by Ensemble despite Diagonal and Mixture both being more
accurate in MSE.

**Multivariate is a clear regression, and the reason is visible in the magnitudes.**
On `KS` its PRR collapses across the board (Epistemic 0.409, Aleatoric 0.208, both far
below Diagonal's 0.845/0.839 and even below Ensemble), and its rank correlation with
the realised error is similarly weak (ρ ≈ 0.13–0.41). On `Burgers` the PRR numbers still
look strong (0.95–0.97, because Burgers is easy enough that most methods stay close to
oracle), but ρ against the *whole* test set craters for Aleatoric/Total (0.29 vs
Diagonal's 0.92) — it only gets the top of the retention curve right. Section 6 explains
why: Multivariate's EU share of TU is 0.87 (`KS`) and 0.73 (`Burgers`), roughly 10–20×
higher than Diagonal's or Mixture's on the same datasets, and its raw AU/EU magnitudes
are an order of magnitude larger too. Two confounds sit underneath this and are not
disentangled here: Multivariate uses a different noise schedule than Diagonal/Mixture on
`KS` (beta endpoints (0.001, 0.35) vs (0.001, 0.2) — see Section 8), and only the
diagonal of its predicted covariance is used, discarding the off-diagonal structure the
head was actually trained to produce. Either could explain the blown-up, less-informative
variance; this experiment cannot separate them.

**Where selective prediction works at all** is unchanged from the two-method version of
this analysis: PRR is high on the structured field problems (`Burgers` ≈ 0.99, `KS` ≈
0.79–0.85, `T2M` 0.73) and on `yacht` (n = 31, so noisy), moderate on `naval`, `energy`,
`protein`, `concrete` (0.24–0.79 depending on method), and weak on `kin8nm`, `power`
(~0.10–0.25) and `wine` (~0.03–0.18). `wine`'s discrete integer quality labels make the
error largely irreducible label noise that no input-dependent uncertainty predicts well;
note its ρ values (0.34–0.57) are much better than its PRR (near zero) — PRR only
integrates retention ≥ 0.5, so a ranking can be broadly sensible and still fail to
isolate the very worst examples.

**Total ≈ Aleatoric for Diagonal and Mixture.** Their EU is 0.4%–11% of TU (Section 6),
so `tu ≈ au`. For Ensemble the split is far more even (EU is 6.5%–82% of TU), and for
Multivariate EU actually *dominates* TU (73–87%) — the opposite pattern, for the reasons
above rather than because Multivariate's epistemic signal is unusually strong.

**Rank agreement with the ensemble (Section 5)** tells a similar story: Diagonal and
Mixture agree with the ensemble's AU ranking fairly well (Spearman mostly 0.39–0.96) and
less consistently on EU (0.15–0.89). Multivariate's agreement with the ensemble is
markedly weaker and more erratic (e.g. `Burgers` AU Spearman 0.28 vs Diagonal's 0.82,
despite a *higher* Pearson of 0.92 — the magnitudes move together loosely but the
per-example order does not), consistent with it being the least reliable of the three
distributional heads for this purpose.

---

## 8. Provenance and caveats

* **EU reduction.** The distributional EU reported here is the **mean over all reverse
  diffusion steps** (`--eu-reduction mean`, all steps) for every head. The script
  supports restricting to the last K low-noise steps or taking the max; those variants
  are saved under suffixed filenames and are **not** in this report.
* **Sample budget.** 100 diffusion samples per test example for every method (ensemble:
  5 members × 20 samples), 50 diffusion timesteps, DDIM, linear noise schedule, seed 1.
* **Noise schedules differ by method and, for Multivariate, by dataset**, because the
  runs were trained that way: Diagonal and Mixture pin beta endpoints (0.001, 0.2) on
  KS/Burgers/T2M; Multivariate uses (0.001, 0.35) on KS and T2M but (0.001, 0.2) on
  Burgers; the deterministic (ensemble) checkpoints never set `beta_endpoints` and fall
  back to the (0.001, 0.35) default. Each checkpoint is sampled with the schedule it was
  trained under — see Section 7 for why this matters for interpreting Multivariate's
  results.
* **Mixture components** (`n_components`) differ per dataset: 50 (KS), 2 (Burgers), 10
  (T2M).
* **Multivariate covariance** is low-rank/LoRA, rank 1, on KS/Burgers/T2M; it was never
  trained on UCI since scalar targets would make it coincide with the Diagonal head.
  Only the diagonal of its predicted covariance enters EU/AU here — off-diagonal
  correlations are ignored (see Section 7).
* **T2M has no ensemble** — only one deterministic checkpoint was ever trained. Its
  Mixture and Multivariate checkpoints exist but have not yet been run through
  `selective_prediction.py`, so T2M currently shows Diagonal only.
* **KS/Burgers checkpoints** are each head's seed-1 run; for KS/Burgers the earliest
  timestamp of each head was taken to match the Diagonal checkpoint.
* **Small test sets.** `yacht` (n = 31), `energy` (n = 77) and `concrete` (n = 103) give
  noisy PRR/ρ estimates; no confidence intervals or seed repetitions were computed, so
  differences on these datasets should be described qualitatively, not as significant.
* **Single split for UCI.** All UCI results are Yarin-Gal split 0 only, not averaged over
  the usual 20 splits. Diagonal, Mixture and all five ensemble members share that split,
  and the split-0 test set is held out from every one of them.
* **One-step predictions.** No autoregressive rollout anywhere in this experiment
  (unlike `evaluation/eu.ipynb`).
* **MSE is in standardised units** and not comparable across datasets.
* All PRR, ρ (rank correlation with error), and ensemble-agreement (Pearson/Spearman)
  values reproduce `evaluation/selective_prediction.ipynb` exactly for the quantities it
  prints (PRR, rank correlation vs. the ensemble); the error-rank-correlation (ρ) and
  magnitude (Section 6) columns are computed from the same saved `.npz` arrays but are
  not themselves printed by the notebook.

### Figures available

Under `evaluation/plots/`: `selective_prediction_panels.pdf` and
`selective_prediction_panels_ensemble.pdf` (retention curves, one panel per dataset),
`<dataset>[_ensemble]_selective_prediction.pdf` (single-panel retention curves), and
`eu_spearman_by_dataset.pdf` (EU rank agreement, sorted bar chart). These predate the
Mixture/Multivariate heads and cover Diagonal vs. Ensemble only — no script currently
regenerates them for the four-method comparison.
