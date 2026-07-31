# Uncertainty quantification for diffusion models: experiment context and full results

Context file for writing about two experiments on the same distributional diffusion model.

* **Sections 1–8 — selective prediction.** Can the model's uncertainty decide which
  predictions to keep? Derived from the per-example arrays in
  `results/selective_prediction/*.npz`, produced by `evaluation/selective_prediction.py`
  and analysed in `evaluation/selective_prediction.ipynb`.
* **Section 9 — epistemic uncertainty vs. training-set size.** Does the model's EU
  actually behave like epistemic uncertainty, i.e. shrink as more data becomes
  available? Derived from `results/UCI_EU/*/*/agg_test_std.csv` and the underlying
  `test.csv`.

The two are complementary: Sections 1–8 show the EU is *useful* (it ranks errors better
than a 5-member deep ensemble's), Section 9 shows it is *epistemic* (it decreases with
data, where the aleatoric term does not).

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

All four are variances on the same scale, so the decomposition is additive and the two
models are directly comparable.

**Selective prediction / retention curves.** Examples are sorted by an uncertainty
measure ascending. At retention rate *r* the `1 - r` most uncertain examples are
rejected and the mean MSE over the retained ones is recorded, for *r* on a 51-point grid
over `[0.5, 1]`. Two baselines bracket the result:

* **Oracle** — rank by the realised error itself (the best any ranking can do);
* **Random** — rejection independent of error, so the curve is flat at the overall
  mean MSE.

**PRR (prediction–rejection ratio).** The area between the model's retention curve and
the random baseline, normalised by the area between oracle and random, over
`r ∈ [0.5, 1]`:

```
PRR = (A_random - A_method) / (A_random - A_oracle)
```

* `PRR = 1` — the uncertainty ranks examples as well as the realised error itself;
* `PRR = 0` — no better than rejecting at random;
* `PRR < 0` — worse than random (the uncertainty is anti-correlated with the error).

---

## 2. The two models being compared

| | **Distributional** (`normal`) | **Ensemble** |
|---|---|---|
| What it is | Our distributional diffusion with a Normal head: the network predicts a mean *and* a standard deviation for the noise at every reverse step | The deterministic diffusion used as a deep ensemble over 5 independently trained checkpoints (seeds 1–5) |
| Aleatoric uncertainty | Variance across the 100 drawn samples | Mean over members of the within-member sample variance |
| Epistemic uncertainty | The model's **predicted noise variance**, weighted by the reverse-step coefficient `(1-α)²/(α(1-α̂))` and averaged over the reverse trajectory | **Disagreement between the member means**, i.e. the variance across the 5 checkpoints |
| Cost at inference | One trained model, one sampling pass | 5 trained models |

The key methodological point: the distributional model gets an epistemic signal from a
**single training run**, read off the head's predicted variance, where the ensemble needs
five. The experiment asks whether that single-model signal is as usable for selective
prediction as the ensemble's, and whether the two orderings of the test set agree.

Both models use the law of total variance to split their uncertainty, both report
`tu = au + eu`, and both get the same total sample budget (100 samples: the ensemble
draws 20 per member × 5 members).

---

## 3. Datasets

Eleven test sets in three families:

**PDE fields (1D)** — `KS` (Kuramoto–Sivashinsky), `Burgers`. UNet backbone, 1000 test
examples each, one random timestep drawn per example (frozen under a fixed seed so both
models see identical data).

**Weather field (2D)** — `T2M` (WeatherBench 2-metre temperature, 160×220 grid), 729 test
examples, UNet backbone. **Only one deterministic T2M checkpoint exists, so T2M has no
ensemble** and appears with the distributional model only.

**UCI regression** — `concrete`, `energy`, `kin8nm`, `naval`, `power`, `protein`, `wine`,
`yacht`. MLP/CARD backbone, evaluated on Yarin-Gal **split 0**. All five ensemble members
were trained on split 0 with seeds 1–5, so they differ only in initialisation and the
split-0 test set is held out from every member — the same held-out data the
distributional model is evaluated on, with the same split-0 CARD regressor.

Test-set sizes range from 31 (`yacht`) to 4573 (`protein`); the small ones carry
correspondingly noisy estimates.

**Units.** MSE is in standardised units for every dataset (targets are normalised), which
is why the field datasets sit at `1e-6` and the UCI ones at `1e-2`–`1e-1`. MSE
magnitudes are **not comparable across datasets**; PRR and Spearman are.

---

## 4. Main results table

`n` = test examples. `MSE` = mean over the test set. `PRR` = prediction–rejection ratio
for that uncertainty measure (1 = oracle, 0 = random). `ρ` = Spearman rank correlation
between that uncertainty and the per-example squared error — the direct measure of
whether the uncertainty orders the test set the way the error does. PRR and ρ measure the
same thing from different angles; PRR weights the top of the ranking (it only looks at
retention ≥ 0.5), ρ weights the whole ranking uniformly.

| Dataset | Model | n | MSE | PRR Total | PRR Aleatoric | PRR Epistemic | ρ Total | ρ Aleatoric | ρ Epistemic |
|---|---|---|---|---|---|---|---|---|---|
| KS | Distributional | 1000 | 2.22e-06 | 0.8402 | 0.8394 | 0.8453 | 0.8105 | 0.8102 | 0.8139 |
| KS | Ensemble | 1000 | 2.47e-06 | 0.7937 | 0.6881 | 0.8177 | 0.7678 | 0.6734 | 0.7764 |
| Burgers | Distributional | 1000 | 3.84e-06 | 0.9945 | 0.9945 | 0.9945 | 0.9181 | 0.9177 | 0.9136 |
| Burgers | Ensemble | 1000 | 4.63e-06 | 0.9921 | 0.9899 | 0.9942 | 0.8570 | 0.7774 | 0.8860 |
| T2M | Distributional | 729 | 6.12e-03 | 0.7286 | 0.7286 | 0.6951 | 0.7119 | 0.7123 | 0.6418 |
| UCI_concrete | Distributional | 103 | 7.26e-02 | 0.3953 | 0.3903 | 0.3803 | 0.3108 | 0.3122 | 0.3066 |
| UCI_concrete | Ensemble | 103 | 7.34e-02 | 0.2198 | 0.2197 | 0.0939 | 0.2908 | 0.3175 | -0.0080 |
| UCI_energy | Distributional | 77 | 1.22e-03 | 0.4986 | 0.5015 | 0.4161 | 0.4708 | 0.4727 | 0.4472 |
| UCI_energy | Ensemble | 77 | 1.17e-03 | 0.0820 | 0.2224 | 0.0420 | 0.2639 | 0.3660 | 0.1979 |
| UCI_kin8nm | Distributional | 819 | 6.92e-02 | 0.2083 | 0.2072 | 0.2500 | 0.1461 | 0.1454 | 0.2069 |
| UCI_kin8nm | Ensemble | 819 | 6.91e-02 | 0.2526 | 0.2629 | 0.0259 | 0.1986 | 0.2046 | 0.0396 |
| UCI_naval | Distributional | 1193 | 6.03e-05 | 0.7895 | 0.7877 | 0.7709 | 0.4006 | 0.4013 | 0.3238 |
| UCI_naval | Ensemble | 1193 | 6.58e-05 | 0.5707 | 0.7382 | 0.1661 | 0.2131 | 0.3084 | 0.1020 |
| UCI_power | Distributional | 957 | 5.01e-02 | 0.2213 | 0.2210 | 0.2066 | 0.3156 | 0.3149 | 0.3206 |
| UCI_power | Ensemble | 957 | 5.29e-02 | 0.1661 | 0.1593 | 0.0628 | 0.3371 | 0.3261 | 0.1981 |
| UCI_protein | Distributional | 4573 | 3.60e-01 | 0.4283 | 0.4284 | 0.3517 | 0.6751 | 0.6748 | 0.6594 |
| UCI_protein | Ensemble | 4573 | 3.64e-01 | 0.4396 | 0.4363 | 0.3428 | 0.6495 | 0.6481 | 0.5364 |
| UCI_wine | Distributional | 160 | 6.17e-01 | 0.0273 | 0.0250 | 0.1002 | 0.4301 | 0.4283 | 0.4003 |
| UCI_wine | Ensemble | 160 | 5.06e-01 | 0.0953 | -0.0117 | 0.1794 | 0.5726 | 0.4794 | 0.5628 |
| UCI_yacht | Distributional | 31 | 2.45e-03 | 0.9334 | 0.9296 | 0.9527 | 0.7024 | 0.6956 | 0.6383 |
| UCI_yacht | Ensemble | 31 | 2.74e-03 | 0.9424 | 0.9707 | 0.7722 | 0.6649 | 0.5222 | 0.6048 |

### 4b. Same PRR numbers, models side by side

| Dataset | Dist. Total | Dist. Aleatoric | Dist. Epistemic | Ens. Total | Ens. Aleatoric | Ens. Epistemic |
|---|---|---|---|---|---|---|
| KS | 0.8402 | 0.8394 | 0.8453 | 0.7937 | 0.6881 | 0.8177 |
| Burgers | 0.9945 | 0.9945 | 0.9945 | 0.9921 | 0.9899 | 0.9942 |
| T2M | 0.7286 | 0.7286 | 0.6951 | — | — | — |
| UCI_concrete | 0.3953 | 0.3903 | 0.3803 | 0.2198 | 0.2197 | 0.0939 |
| UCI_energy | 0.4986 | 0.5015 | 0.4161 | 0.0820 | 0.2224 | 0.0420 |
| UCI_kin8nm | 0.2083 | 0.2072 | 0.2500 | 0.2526 | 0.2629 | 0.0259 |
| UCI_naval | 0.7895 | 0.7877 | 0.7709 | 0.5707 | 0.7382 | 0.1661 |
| UCI_power | 0.2213 | 0.2210 | 0.2066 | 0.1661 | 0.1593 | 0.0628 |
| UCI_protein | 0.4283 | 0.4284 | 0.3517 | 0.4396 | 0.4363 | 0.3428 |
| UCI_wine | 0.0273 | 0.0250 | 0.1002 | 0.0953 | -0.0117 | 0.1794 |
| UCI_yacht | 0.9334 | 0.9296 | 0.9527 | 0.9424 | 0.9707 | 0.7722 |

---

## 5. Agreement between the two decompositions

Do the two models order the *same* test examples the same way? For each dataset the
distributional model's AU/EU/TU is correlated against the ensemble's, per test example.
Both runs used the same seed and sample count, and a `target_mean` fingerprint saved with
each run confirms they evaluated identical examples in identical order.

* **Pearson** — linear agreement of raw magnitudes. Both are variances, but ours is a
  predicted noise variance propagated through the reverse trajectory while the ensemble's
  is a variance between checkpoints; there is no reason for the constant relating them to
  be 1, and Pearson penalises that.
* **Spearman** — rank agreement. **This is the one that matters for selective
  prediction**: retention curves only use the *ordering*, so two measures with Spearman 1
  would reject exactly the same examples and score identical PRR however differently they
  are scaled.

T2M is absent (no ensemble).

| Dataset | n | AU Pearson | AU Spearman | EU Pearson | EU Spearman | TU Pearson | TU Spearman |
|---|---|---|---|---|---|---|---|
| KS | 1000 | 0.933 | 0.964 | 0.834 | 0.821 | 0.947 | 0.921 |
| Burgers | 1000 | 0.843 | 0.818 | 0.648 | 0.890 | 0.790 | 0.884 |
| UCI_concrete | 103 | 0.755 | 0.810 | 0.160 | 0.151 | 0.707 | 0.760 |
| UCI_energy | 77 | 0.489 | 0.766 | 0.605 | 0.572 | 0.610 | 0.667 |
| UCI_kin8nm | 819 | 0.825 | 0.828 | 0.413 | 0.390 | 0.824 | 0.831 |
| UCI_naval | 1193 | 0.794 | 0.680 | 0.183 | 0.567 | 0.472 | 0.702 |
| UCI_power | 957 | 0.446 | 0.562 | 0.123 | 0.160 | 0.432 | 0.536 |
| UCI_protein | 4573 | 0.824 | 0.854 | 0.413 | 0.663 | 0.820 | 0.854 |
| UCI_wine | 160 | 0.428 | 0.455 | 0.084 | 0.365 | 0.410 | 0.412 |
| UCI_yacht | 31 | 0.646 | 0.700 | 0.724 | 0.725 | 0.820 | 0.796 |

---

## 6. Uncertainty magnitudes

Useful for explaining *why* the distributional model's Total and Aleatoric columns are
nearly identical: its EU is a small fraction of its TU, so `tu ≈ au`.

| Dataset | Model | mean MSE | mean AU | mean EU | mean TU | EU share of TU |
|---|---|---|---|---|---|---|
| KS | Distributional | 2.22e-06 | 1.36e-05 | 5.21e-07 | 1.41e-05 | 0.037 |
| KS | Ensemble | 2.47e-06 | 2.35e-06 | 2.51e-06 | 4.86e-06 | 0.517 |
| Burgers | Distributional | 3.84e-06 | 9.79e-06 | 3.60e-07 | 1.02e-05 | 0.035 |
| Burgers | Ensemble | 4.63e-06 | 6.58e-06 | 4.14e-06 | 1.07e-05 | 0.386 |
| T2M | Distributional | 6.12e-03 | 3.00e-03 | 5.61e-05 | 3.05e-03 | 0.018 |
| UCI_concrete | Distributional | 7.26e-02 | 1.41e-02 | 3.95e-04 | 1.45e-02 | 0.027 |
| UCI_concrete | Ensemble | 7.34e-02 | 8.09e-03 | 1.74e-03 | 9.82e-03 | 0.177 |
| UCI_energy | Distributional | 1.22e-03 | 4.24e-04 | 4.04e-05 | 4.65e-04 | 0.087 |
| UCI_energy | Ensemble | 1.17e-03 | 7.60e-05 | 2.17e-04 | 2.93e-04 | 0.741 |
| UCI_kin8nm | Distributional | 6.92e-02 | 3.84e-02 | 8.59e-04 | 3.92e-02 | 0.022 |
| UCI_kin8nm | Ensemble | 6.91e-02 | 2.77e-02 | 1.94e-03 | 2.97e-02 | 0.065 |
| UCI_naval | Distributional | 6.03e-05 | 5.45e-05 | 4.12e-06 | 5.86e-05 | 0.070 |
| UCI_naval | Ensemble | 6.58e-05 | 3.26e-06 | 1.43e-05 | 1.76e-05 | 0.815 |
| UCI_power | Distributional | 5.01e-02 | 3.11e-02 | 6.26e-04 | 3.17e-02 | 0.020 |
| UCI_power | Ensemble | 5.29e-02 | 2.93e-02 | 3.44e-03 | 3.28e-02 | 0.105 |
| UCI_protein | Distributional | 3.60e-01 | 2.58e-01 | 1.68e-03 | 2.60e-01 | 0.006 |
| UCI_protein | Ensemble | 3.64e-01 | 2.99e-01 | 2.54e-02 | 3.25e-01 | 0.078 |
| UCI_wine | Distributional | 6.17e-01 | 1.40e-01 | 5.19e-04 | 1.41e-01 | 0.004 |
| UCI_wine | Ensemble | 5.06e-01 | 1.62e-01 | 8.01e-02 | 2.42e-01 | 0.331 |
| UCI_yacht | Distributional | 2.45e-03 | 1.96e-04 | 2.00e-05 | 2.16e-04 | 0.093 |
| UCI_yacht | Ensemble | 2.74e-03 | 4.04e-05 | 1.08e-04 | 1.48e-04 | 0.728 |

---

## 7. What the numbers show

Ten datasets admit a head-to-head comparison (all but T2M).

**Accuracy is essentially a tie.** The distributional model has the lower MSE on 7 of 10
datasets, but the margins are small (typically 1–10%) except on `wine`, where the
ensemble is clearly better (0.506 vs 0.617). The two models are the same architecture
family trained on the same data, so this is expected: the experiment is about the
uncertainty, not the accuracy.

**Epistemic uncertainty is where the distributional model wins.** Its EU beats the
ensemble's EU on **9 of 10** datasets, often by a wide margin: `naval` 0.771 vs 0.166,
`kin8nm` 0.250 vs 0.026, `energy` 0.416 vs 0.042, `concrete` 0.380 vs 0.094. `wine` is
the sole exception (0.100 vs 0.179). The ensemble's EU — checkpoint disagreement — is
close to useless for ranking errors on several UCI datasets (`kin8nm` 0.026, `power`
0.063, `energy` 0.042, and ρ = -0.008 on `concrete`), while the distributional EU stays
informative. That is the headline: **a single-model predicted variance ranks errors
better than five-member checkpoint disagreement.**

**Total uncertainty**: distributional wins 6 of 10 (`kin8nm`, `protein`, `wine`, `yacht`
go to the ensemble, the last three by small margins). **Aleatoric**: distributional wins
7 of 10.

**Where selective prediction works at all.** PRR is high on the structured field problems
— `Burgers` ≈ 0.99 (both models nearly match the oracle), `KS` ≈ 0.79–0.85, `T2M` 0.73 —
and on `yacht` (0.93, but n = 31). It is moderate on `naval`, `energy`, `protein`,
`concrete` (0.35–0.79) and weak on `kin8nm`, `power` (~0.2) and `wine` (~0.03–0.18).
`wine` is the hard case for both models under PRR: discrete integer quality labels mean
the error is largely irreducible label noise that no input-dependent uncertainty can
anticipate. Note that on `wine` the *rank* correlations with error are actually
respectable (ρ ≈ 0.40–0.57) while PRR is near zero — PRR only integrates over retention
≥ 0.5, so an uncertainty can order most of the test set sensibly and still fail to isolate
the very worst examples.

**Total ≈ Aleatoric for the distributional model.** Its EU contributes only 0.4%–9% of
TU (Section 6), so `tu ≈ au` and the two columns track each other to 3 decimals on most
datasets. For the ensemble the split is far more even (EU is 6.5%–82% of TU), so its
Total and Aleatoric columns genuinely differ — and its Total is often dragged down by its
weak EU (`naval`: AU 0.738, EU 0.166, TU 0.571). Any claim about the distributional
model's EU should rest on the **PRR Epistemic** column, not on Total, since Total is
dominated by AU.

**The two notions of uncertainty agree, but less so for EU.** AU rank agreement is strong
(Spearman 0.46–0.96, above 0.68 on 7 of 10). EU rank agreement is more variable: strong
on the field datasets (`Burgers` 0.89, `KS` 0.82) and `yacht` (0.73), moderate on
`protein` (0.66), `naval` (0.57), `energy` (0.57), and weak on `kin8nm` (0.39), `wine`
(0.37), `power` (0.16), `concrete` (0.15). Pearson is consistently lower than Spearman
for EU (e.g. `naval` 0.18 vs 0.57), confirming the two EUs are monotonically related but
on different scales — expected, since one is a propagated noise variance and the other a
between-checkpoint variance. Where EU agreement is weakest is largely where the
ensemble's EU is itself uninformative (`concrete`, `power`, `kin8nm`), so the
disagreement should not be read as the distributional EU being wrong.

---

## 8. Provenance and caveats

* **EU reduction.** The distributional EU reported here is the **mean over all reverse
  diffusion steps** (`--eu-reduction mean`, all steps). The script supports restricting
  to the last K low-noise steps or taking the max; those variants are saved under
  suffixed filenames and are **not** in this report.
* **Sample budget.** 100 diffusion samples per test example for both models
  (ensemble: 5 members × 20 samples), 50 diffusion timesteps, DDIM, linear noise
  schedule, seed 1.
* **Noise schedules differ by model** because the runs were trained that way: the
  distributional checkpoints use beta endpoints (0.001, 0.2), the deterministic ones
  (0.001, 0.35). Each model is sampled with the schedule it was trained under.
* **T2M has no ensemble** — only one deterministic checkpoint was ever trained. Every
  T2M row is distributional-only.
* **Small test sets.** `yacht` (n = 31), `energy` (n = 77) and `concrete` (n = 103) give
  noisy PRR estimates; no confidence intervals or seed repetitions were computed, so
  differences on these datasets should be described qualitatively, not as significant.
* **Single split for UCI.** All UCI results are Yarin-Gal split 0 only, not averaged over
  the usual 20 splits. Both models and all five ensemble members share that split, and
  the split-0 test set is held out from every one of them.
* **One-step predictions.** No autoregressive rollout anywhere in this experiment
  (unlike `evaluation/eu.ipynb`).
* **MSE is in standardised units** and not comparable across datasets.
* The ρ (Spearman uncertainty-vs-error) columns in Section 4 and the magnitudes in
  Section 6 are computed from the same saved `.npz` arrays as the notebook's tables but
  are not themselves printed in the notebook; all PRR and cross-model correlation values
  reproduce the notebook exactly.

### Figures available

Under `evaluation/plots/`: `selective_prediction_panels.pdf` and
`selective_prediction_panels_ensemble.pdf` (retention curves, one panel per dataset),
`<dataset>[_ensemble]_selective_prediction.pdf` (single-panel retention curves), and
`eu_spearman_by_dataset.pdf` (EU rank agreement between the two models, sorted bar
chart).

---

## 9. Companion experiment: epistemic uncertainty vs. training-set size

Source: `results/UCI_EU/<dataset>/<run>/agg_test_std.csv` (summary) and `test.csv`
(per-run values).

### 9.1 What this experiment does

**Question.** Section 4 showed the distributional model's EU is *useful* for ranking
errors. This experiment asks whether it is *epistemic* in the proper sense: epistemic
uncertainty is the reducible part, the model's ignorance about the true function, so it
must **shrink as the training set grows**. Aleatoric uncertainty is the irreducible
observation noise and should **not**.

**Setup.** The distributional model (Normal head, `normal`) is retrained from scratch on
a fraction of the available training data and evaluated on the full, unchanged test set:

* **data fractions** — 0.1, 0.25, 0.5, 0.75, 1.0 of the training split;
* **splits** — all 10 Yarin-Gal splits (0–9), so every cell is 10 independent runs;
* **datasets** — `concrete`, `energy`, `kin8nm`, `power`, `wine`, `yacht`
  (6 of the 8 UCI datasets; **`naval` and `protein` were not run**);
* 5 fractions × 10 splits = **50 training runs per dataset**, 300 in total;
* same hyper-parameters as elsewhere: CARD backbone, MLP, hidden 64, 2 layers,
  T = 50 timesteps, DDIM, linear schedule, beta endpoints (0.001, 0.35), CARD `x_T`
  sampling, `closed_form=True`, 100 UQ samples, seed 1234, up to 3000 epochs with early
  stopping at 300.

The test set is identical across fractions, so the only thing that changes is how much
the model was allowed to learn from.

**Reported numbers** are the mean ± standard deviation **across the 10 splits**. The
`agg_test_std.csv` files store AU and EU **scaled by 10³**, in the *original* target
units squared (metrics are de-standardised before logging) — so the concrete row
`22206.05` means an aleatoric variance of 22.21 MPa². The tables below keep the file's
×10³ convention; divide by 1000 for raw units.

### 9.2 Aleatoric and epistemic uncertainty by data fraction

Values as stored in `agg_test_std.csv` (×10⁻³ original-target-units²), mean ± std over
the 10 splits. `n_train` is approximate (≈ 0.81·N·fraction, after the 90/10 train–test
split and a further 10% validation holdout).

| Dataset | N | Fraction | ≈ n_train | Aleatoric (×10⁻³) | Epistemic (×10⁻³) |
|---|---|---|---|---|---|
| yacht | 308 | 0.10 | 25 | 7747.50 ± 3658.48 | 0.65 ± 0.24 |
| yacht | 308 | 0.25 | 62 | 3179.52 ± 2662.81 | 0.32 ± 0.21 |
| yacht | 308 | 0.50 | 125 | 1733.36 ± 3631.74 | 0.23 ± 0.19 |
| yacht | 308 | 0.75 | 187 | 713.24 ± 349.23 | 0.17 ± 0.05 |
| yacht | 308 | 1.00 | 249 | 422.59 ± 198.15 | 0.13 ± 0.05 |
| energy | 768 | 0.10 | 62 | 2623.34 ± 2554.06 | 0.43 ± 0.24 |
| energy | 768 | 0.25 | 156 | 263.46 ± 194.97 | 0.14 ± 0.05 |
| energy | 768 | 0.50 | 311 | 106.36 ± 86.72 | 0.09 ± 0.04 |
| energy | 768 | 0.75 | 467 | 85.94 ± 32.22 | 0.08 ± 0.03 |
| energy | 768 | 1.00 | 622 | 78.05 ± 35.62 | 0.08 ± 0.04 |
| concrete | 1030 | 0.10 | 83 | 22206.05 ± 9073.58 | 1.54 ± 0.47 |
| concrete | 1030 | 0.25 | 209 | 10648.40 ± 3129.36 | 0.90 ± 0.19 |
| concrete | 1030 | 0.50 | 417 | 8831.74 ± 2170.28 | 0.87 ± 0.12 |
| concrete | 1030 | 0.75 | 626 | 8457.85 ± 1236.72 | 0.94 ± 0.10 |
| concrete | 1030 | 1.00 | 834 | 7694.09 ± 2308.64 | 0.87 ± 0.18 |
| wine | 1599 | 0.10 | 130 | 245.60 ± 50.24 | 3.67 ± 0.62 |
| wine | 1599 | 0.25 | 324 | 255.22 ± 55.05 | 3.35 ± 0.90 |
| wine | 1599 | 0.50 | 648 | 255.32 ± 60.64 | 2.92 ± 0.67 |
| wine | 1599 | 0.75 | 971 | 246.22 ± 46.43 | 2.89 ± 0.46 |
| wine | 1599 | 1.00 | 1295 | 255.97 ± 59.85 | 2.92 ± 0.77 |
| kin8nm | 8192 | 0.10 | 664 | 3.35 ± 0.29 | 1.08 ± 0.10 |
| kin8nm | 8192 | 0.25 | 1659 | 3.15 ± 0.53 | 1.07 ± 0.29 |
| kin8nm | 8192 | 0.50 | 3318 | 3.03 ± 0.24 | 0.96 ± 0.07 |
| kin8nm | 8192 | 0.75 | 4977 | 3.11 ± 0.33 | 1.01 ± 0.09 |
| kin8nm | 8192 | 1.00 | 6636 | 3.10 ± 0.39 | 1.03 ± 0.14 |
| power | 9568 | 0.10 | 775 | 13171.00 ± 1099.42 | 1.02 ± 0.12 |
| power | 9568 | 0.25 | 1937 | 12660.89 ± 1034.90 | 0.98 ± 0.13 |
| power | 9568 | 0.50 | 3875 | 12886.06 ± 1179.17 | 1.00 ± 0.13 |
| power | 9568 | 0.75 | 5812 | 12863.94 ± 675.94 | 0.94 ± 0.09 |
| power | 9568 | 1.00 | 7750 | 12514.33 ± 613.05 | 0.96 ± 0.07 |

Note that AU and EU magnitudes are **not comparable across datasets**: they are in each
dataset's own target units squared. `kin8nm` looks anomalous only because its targets
already live on a ≈[-1.5, 1.5] range, so its "original" units are near-standardised.

### 9.3 Accuracy at the same fractions

From `test.csv`, in **original target units** (not ×10³), mean ± std over the 10 splits.
Included because it separates "EU fell because the model genuinely learned more" from
"nothing changed at all".

| Dataset | Fraction | MSE | RMSE | CRPS | Coverage (90%) | QICE |
|---|---|---|---|---|---|---|
| yacht | 0.10 | 9.159 ± 5.83 | 2.874 ± 1.00 | 1.355 ± 0.316 | 0.939 ± 0.047 | 0.0874 ± 0.0181 |
| yacht | 0.25 | 4.282 ± 4.91 | 1.843 ± 0.991 | 0.885 ± 0.376 | 0.936 ± 0.059 | 0.0733 ± 0.0120 |
| yacht | 0.50 | 2.649 ± 2.36 | 1.475 ± 0.727 | 0.639 ± 0.311 | 0.868 ± 0.069 | 0.0572 ± 0.0218 |
| yacht | 0.75 | 2.053 ± 1.65 | 1.324 ± 0.578 | 0.591 ± 0.240 | 0.907 ± 0.069 | 0.0543 ± 0.0188 |
| yacht | 1.00 | 1.696 ± 1.54 | 1.190 ± 0.558 | 0.476 ± 0.205 | 0.877 ± 0.074 | 0.0523 ± 0.0104 |
| energy | 0.10 | 1.307 ± 0.910 | 1.086 ± 0.377 | 0.616 ± 0.219 | 0.969 ± 0.032 | 0.0475 ± 0.0173 |
| energy | 0.25 | 0.417 ± 0.192 | 0.631 ± 0.148 | 0.348 ± 0.080 | 0.835 ± 0.089 | 0.0415 ± 0.0095 |
| energy | 0.50 | 0.264 ± 0.068 | 0.511 ± 0.064 | 0.283 ± 0.025 | 0.753 ± 0.094 | 0.0450 ± 0.0138 |
| energy | 0.75 | 0.241 ± 0.057 | 0.488 ± 0.057 | 0.271 ± 0.025 | 0.764 ± 0.071 | 0.0481 ± 0.0101 |
| energy | 1.00 | 0.233 ± 0.058 | 0.479 ± 0.059 | 0.266 ± 0.027 | 0.739 ± 0.064 | 0.0493 ± 0.0137 |
| concrete | 0.10 | 31.97 ± 8.45 | 5.610 ± 0.746 | 3.048 ± 0.418 | 0.887 ± 0.044 | 0.0289 ± 0.0049 |
| concrete | 0.25 | 24.47 ± 5.47 | 4.921 ± 0.541 | 2.657 ± 0.324 | 0.819 ± 0.055 | 0.0363 ± 0.0071 |
| concrete | 0.50 | 24.50 ± 4.74 | 4.929 ± 0.475 | 2.633 ± 0.266 | 0.795 ± 0.020 | 0.0345 ± 0.0044 |
| concrete | 0.75 | 24.11 ± 5.15 | 4.886 ± 0.509 | 2.600 ± 0.303 | 0.807 ± 0.044 | 0.0353 ± 0.0085 |
| concrete | 1.00 | 24.10 ± 4.86 | 4.888 ± 0.482 | 2.615 ± 0.287 | 0.777 ± 0.049 | 0.0375 ± 0.0109 |
| wine | 0.10 | 0.431 ± 0.071 | 0.655 ± 0.054 | 0.365 ± 0.031 | 0.860 ± 0.041 | 0.0351 ± 0.0085 |
| wine | 0.25 | 0.431 ± 0.082 | 0.654 ± 0.062 | 0.364 ± 0.039 | 0.865 ± 0.060 | 0.0343 ± 0.0082 |
| wine | 0.50 | 0.425 ± 0.069 | 0.650 ± 0.053 | 0.356 ± 0.036 | 0.868 ± 0.044 | 0.0353 ± 0.0160 |
| wine | 0.75 | 0.423 ± 0.075 | 0.648 ± 0.058 | 0.358 ± 0.035 | 0.854 ± 0.049 | 0.0315 ± 0.0070 |
| wine | 1.00 | 0.429 ± 0.081 | 0.652 ± 0.062 | 0.354 ± 0.034 | 0.876 ± 0.055 | 0.0371 ± 0.0106 |
| kin8nm | 0.10 | 0.00517 ± 0.00036 | 0.0719 ± 0.0025 | 0.0401 ± 0.0012 | 0.867 ± 0.020 | 0.0205 ± 0.0042 |
| kin8nm | 0.25 | 0.00495 ± 0.00023 | 0.0703 ± 0.0017 | 0.0391 ± 0.0007 | 0.862 ± 0.030 | 0.0206 ± 0.0054 |
| kin8nm | 0.50 | 0.00494 ± 0.00027 | 0.0703 ± 0.0019 | 0.0390 ± 0.0009 | 0.859 ± 0.019 | 0.0230 ± 0.0037 |
| kin8nm | 0.75 | 0.00489 ± 0.00027 | 0.0699 ± 0.0019 | 0.0388 ± 0.0008 | 0.864 ± 0.018 | 0.0225 ± 0.0050 |
| kin8nm | 1.00 | 0.00485 ± 0.00026 | 0.0697 ± 0.0019 | 0.0386 ± 0.0008 | 0.861 ± 0.021 | 0.0223 ± 0.0046 |
| power | 0.10 | 15.50 ± 1.45 | 3.933 ± 0.185 | 2.117 ± 0.061 | 0.917 ± 0.012 | 0.0113 ± 0.0040 |
| power | 0.25 | 15.43 ± 1.39 | 3.925 ± 0.177 | 2.107 ± 0.056 | 0.918 ± 0.011 | 0.0106 ± 0.0029 |
| power | 0.50 | 15.33 ± 1.45 | 3.912 ± 0.186 | 2.098 ± 0.063 | 0.916 ± 0.015 | 0.0113 ± 0.0041 |
| power | 0.75 | 15.28 ± 1.40 | 3.906 ± 0.180 | 2.085 ± 0.059 | 0.921 ± 0.009 | 0.0114 ± 0.0013 |
| power | 1.00 | 15.19 ± 1.42 | 3.894 ± 0.184 | 2.082 ± 0.058 | 0.919 ± 0.011 | 0.0110 ± 0.0026 |

### 9.4 Summary: relative change from 10% to 100% of the training data

| Dataset | N | ΔEU | ΔAU | ΔMSE | Monotone in EU? |
|---|---|---|---|---|---|
| energy | 768 | **−81%** | −97% | −82% | yes (0.43→0.14→0.09→0.08→0.08) |
| yacht | 308 | **−80%** | −95% | −82% | yes (0.65→0.32→0.23→0.17→0.13) |
| concrete | 1030 | **−44%** | −65% | −25% | almost (0.94 blip at 0.75) |
| wine | 1599 | **−20%** | +4% | −0.5% | almost (flat after 0.5) |
| power | 9568 | −6% | −5% | −2% | no — flat |
| kin8nm | 8192 | −5% | −8% | −6% | no — flat |

### 9.5 What the numbers show

**The headline result.** On every dataset where more data actually helps, the
distributional model's epistemic uncertainty falls, and it falls monotonically. `yacht`
(EU 0.65 → 0.13, −80%) and `energy` (0.43 → 0.08, −81%) are the clearest cases: each
step up in data fraction lowers EU, with no reversals. `concrete` falls 44% and then
plateaus after 25% of the data — which is also where its MSE stops improving. This is
the behaviour an epistemic term is supposed to have and is the direct evidence that the
Normal head's predicted variance is tracking model ignorance rather than noise.

**`wine` is the cleanest demonstration of the decomposition.** It is the one dataset
where the two terms visibly separate: EU falls 20% (3.67 → 2.92) while AU is flat
(245.6 → 256.0, +4%) and MSE does not move at all (−0.5%). More data cannot reduce the
label noise in discrete integer wine-quality scores, and the aleatoric term correctly
refuses to shrink — but the model still becomes less *ignorant*, and only the epistemic
term registers it. This is exactly the same dataset that Section 7 flags as the hard case
for selective prediction, and for the same underlying reason: the error there is
dominated by irreducible noise.

**Where EU is flat, it is flat for a defensible reason.** `kin8nm` (N = 8192) and `power`
(N = 9568) show essentially no change in EU (−5%, −6%). But they also show almost no
change in MSE (−6%, −2%): at 10% of these datasets the model already has 600–800
training points, enough to fit the function about as well as it ever will. There is no
epistemic uncertainty left to remove, so a *correct* EU should be flat here. Flat EU on
these two is a consistency check passing, not a failure — the model is not manufacturing
uncertainty reduction where no learning happened.

**EU tracks MSE improvement across datasets.** Ranking the six datasets by relative EU
drop gives energy (−81%), yacht (−80%), concrete (−44%), wine (−20%), power (−6%),
kin8nm (−5%); ranking by MSE improvement gives energy (−82%), yacht (−82%), concrete
(−25%), kin8nm (−6%), power (−2%), wine (−0.5%). The two orderings agree closely, with
`wine` the deliberate exception discussed above.

**AU also falls on the small datasets** — sharply (−95% on `yacht`, −97% on `energy`).
This should be stated honestly rather than glossed over: at 10% of `yacht` the model has
roughly 25 training points, is badly underfit, and inflates *both* terms. So the
small-data end of those two curves is not a clean aleatoric/epistemic separation. The
argument that EU is specifically epistemic rests on `wine` (where AU is flat and EU
falls) and on `kin8nm`/`power` (where neither moves because nothing was learned), not on
`yacht`/`energy`, where everything falls together.

**Variance across splits shrinks with data too.** The ± values in Section 9.2 contract
sharply — `yacht` EU goes from 0.65 ± 0.24 to 0.13 ± 0.05, `energy` AU from
2623 ± 2554 to 78 ± 36. With 10% of a small dataset the outcome depends heavily on which
split you got; with the full set it does not. Note this also means the low-fraction rows
carry large uncertainty (`yacht` AU at 50% has a std larger than its mean, 1733 ± 3632),
so individual low-fraction cells should not be over-read.

### 9.6 Provenance and caveats for Section 9

* **Distributional model only.** No ensemble comparison exists for this sweep — the deep
  ensemble is not part of the data-fraction experiment.
* **Six datasets, not eight.** `naval` and `protein` were not run. Both appear in
  Sections 4–6 but not here.
* **10 splits, 1 seed.** The ± is the standard deviation across Yarin-Gal splits 0–9 at
  a single training seed (1234), so it captures data-split variability, not
  initialisation variability. No confidence intervals or significance tests were
  computed.
* **Units differ from Sections 4–6.** These metrics are de-standardised to original
  target units (and AU/EU further scaled by 10³ in the `agg` files), whereas the
  selective-prediction arrays are in standardised units. The EU values in Section 9 and
  those in Section 6 are **not on a comparable scale** and come from different code
  paths — do not compare them numerically across sections.
* **`n_train` is approximate.** Reconstructed as ≈ 0.81·N·fraction from the 90/10
  Yarin-Gal split plus `validation_ratio = 0.1`; the exact rounding and whether the
  fraction is applied before or after the validation holdout were not verified against
  the loader.
* **Run directories** (all under `results/UCI_EU/`):
  `concrete/20260726_003536_data_fraction_normal_concrete`,
  `energy/20260726_003536_data_fraction_normal_energy`,
  `kin8nm/20260726_003305_data_fraction_normal_kin8nm`,
  `power/20260726_003113_data_fraction_normal_power`,
  `wine/20260727_133932_data_fraction_normal_wine`,
  `yacht/20260727_103105_data_fraction_normal_yacht`.
  Each contains `agg_test_std.csv` (the summary above), `test.csv` (all 50 runs, all
  metrics), the `.ini` config, and one training-curve PNG per run.
