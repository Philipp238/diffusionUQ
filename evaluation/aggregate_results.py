# Process experiment results and aggregate into a single table.

import os
import glob
import pandas as pd
import numpy as np


def process_experiment(results_dir: str, experiment: str, models: list, factor: int = 1, agg_groups: list = ["distributional_method"]) -> pd.DataFrame:
    train_metrics = ['t_training_avg', 't_training_med', 't_training_std']
    test_metrics = ['RMSETest', 'EnergyScoreTest', 'CRPSTest', 'Gaussian NLLTest',
                    'C95Test', 'C90Test', 'C75Test', 'C50Test']
    metrics = train_metrics + test_metrics
    factor_metrics = ['RMSETest', 'EnergyScoreTest', 'CRPSTest']

    results = pd.DataFrame()
    for model in models:
        path = f"{results_dir}/{experiment}/{model}/"
        test_files = sorted([f for f in glob.iglob(path + "**", recursive=True)
                             if os.path.isfile(f) and f.endswith("test.csv")])
        for test_file in test_files:
            test_df = pd.read_csv(test_file, index_col=0)
            train_file = os.path.join(os.path.dirname(test_file), "train.csv")
            if os.path.exists(train_file):
                train_df = pd.read_csv(train_file, index_col=0)
                for m in train_metrics:
                    if m in train_df.index:
                        test_df.loc[m] = train_df.loc[m]
                    else:
                        test_df.loc[m] = -1
            else:
                for m in train_metrics:
                    test_df.loc[m] = -1
            results = pd.concat([results, test_df], axis=1)

    rows = metrics.copy()
    for g in agg_groups:
        rows.append(g)
    results = results.loc[rows]
    results.loc[metrics] = results.loc[metrics].astype("float32")
    # Multiply by factor
    results.loc[factor_metrics] = results.loc[factor_metrics] * factor
    results = results.transpose()
    # Turn std into var
    results["t_training_std"] = results["t_training_std"] ** 2
    # Group by uncertainty quantification method
    mean = results.groupby(agg_groups).mean().astype("float32")
    mean.insert(0, "Statistic", "Mean")
    var = results.groupby(agg_groups).var().astype("float32")
    var.insert(0, "Statistic", "Std")
    # Adjust for intra variance
    var["t_training"] = mean["t_training_std"] + var["t_training_avg"]
    var.loc[:, metrics] = np.sqrt(var.loc[:, metrics])
    var.loc[:, "t_training"] = np.sqrt(var.loc[:, "t_training"])
    var.drop(["t_training_avg", "t_training_std"], axis=1, inplace=True)
    mean = mean.rename({"t_training_avg": "t_training"}, axis=1).drop("t_training_std", axis=1)
    results_df = pd.concat([mean.transpose(), var.transpose()], axis=1)
    results_df = results_df[results_df.columns.sort_values().unique()]
    return results_df


def create_latex_table(results_df: pd.DataFrame, results_dir: str, experiment: str, model: str) -> str:
    # Initialize an empty DataFrame to store the formatted values
    formatted_df = pd.DataFrame()
    # Test metrics are everything except the training time row
    metrics = [idx for idx in results_df.index if idx != "Statistic" and idx != "t_training" and idx != "t_training_med"]
    methods = list(results_df.columns.unique())  # Methods are the top level of the columns MultiIndex

    # Create a new DataFrame with the method as the index and metrics as columns
    for metric in metrics:
        formatted_df[metric] = [
            f"\\makecell{{{results_df.loc[metric, method].values[0]:.2f} \\\\ ($\\pm$ {results_df.loc[metric, method].values[1]:.2f})}}"
            for method in methods
        ]
    formatted_df.index = methods
    latex_table = formatted_df.to_latex(escape=False)
    with open(f"{results_dir}/{experiment}/aggregated_results.tex", "w") as f:
        f.write(latex_table)

    # Second table with mean only
    for metric in metrics:
        formatted_df[metric] = [
            f"{results_df.loc[metric, method].values[0]:.2f}"
            for method in methods
        ]
    formatted_df.index = methods
    latex_table = formatted_df.to_latex(escape=False)
    with open(f"{results_dir}/{experiment}/aggregated_results_mean.tex", "w") as f:
        f.write(latex_table)


UNIFIED_METRICS = [
    "MSETest", "RMSETest", "EnergyScoreTest", "CRPSTest",
    "Gaussian NLLTest", "C95Test", "C90Test", "C75Test", "C50Test",
    "MSEValidation", "RMSEValidation", "EnergyScoreValidation",
    "CRPSValidation", "Gaussian NLLValidation",
    "C95Validation", "C90Validation", "C75Validation", "C50Validation",
    "NumberParameters",
]
SINGLE_METHOD_EXPERIMENTS = {"NDP", "crps_ensemble", "UCI_iDDPM"}
SEED_EXPERIMENTS = {"Burgers", "KS", "T2M"}
EXCLUDED_PATH_PARTS = {"hparams", "failed", "noise_tuning", "timing", "archive", "normal_log"}


def discover_runs(results_dir: str = "results", experiments: list | None = None) -> pd.DataFrame:
    top = experiments or ["Burgers", "KS", "T2M", "UCI", "UCI_iDDPM", "NDP", "crps_ensemble"]
    rows = []
    for exp in top:
        root = os.path.join(results_dir, exp)
        if not os.path.isdir(root):
            continue
        for test_file in sorted(glob.iglob(f"{root}/**/test.csv", recursive=True)):
            if EXCLUDED_PATH_PARTS.intersection(test_file.split(os.sep)):
                continue
            df = pd.read_csv(test_file, index_col=0)
            wide = df.T.reset_index(drop=True)
            wide["experiment"] = exp
            wide["source_file"] = os.path.relpath(test_file, results_dir)
            if exp in SINGLE_METHOD_EXPERIMENTS:
                wide["method"] = exp
            elif "distributional_method" in wide.columns:
                wide["method"] = wide["distributional_method"]
            else:
                wide["method"] = exp
            if exp in SEED_EXPERIMENTS and "seed" in wide.columns:
                wide["run_id"] = wide["seed"].astype(str)
            elif "yarin_gal_uci_split_indices" in wide.columns:
                wide["run_id"] = wide["yarin_gal_uci_split_indices"].astype(str)
            else:
                wide["run_id"] = pd.Series(range(len(wide))).astype(str)
            wide["dataset"] = wide["dataset_name"] if "dataset_name" in wide.columns else exp
            rows.append(wide)
    return pd.concat(rows, ignore_index=True)


def aggregate_runs(long_df: pd.DataFrame) -> pd.DataFrame:
    cols = [c for c in UNIFIED_METRICS if c in long_df.columns]
    num = long_df[cols].apply(pd.to_numeric, errors="coerce")
    keyed = pd.concat([long_df[["experiment", "dataset", "method"]], num], axis=1)
    grouped = keyed.groupby(["experiment", "dataset", "method"])
    agg = grouped.agg(["mean", "std"])
    agg.columns = [f"{m}_{s}" for m, s in agg.columns]
    agg["n_runs"] = grouped.size()
    return agg.reset_index()


METHOD_SYMBOL = {
    "deterministic": r"$\bm\delta_\theta$",
    "normal": r"$\bm\Sigma_\theta^{\mathrm{diag}}$",
    "mixednormal": r"$\bm\Sigma_\theta^{\mathrm{mix}}$",
    "mvnormal": r"$\bm\Sigma_\theta^{\mathrm{mv}}$",
    "sample": r"$\bm\epsilon_t^\mathrm{ES}$",
    "iDDPM": "iDDPM",
    "crps_ensemble": "CRPS-Ens.",
    "NDP": "NDP",
}
UCI_DATASET_ORDER = [
    ("concrete", "Concrete"),
    ("energy", "Energy"),
    ("kin8nm", "Kin8nm"),
    ("naval-propulsion-plant", "Naval"),
    ("power-plant", "Power"),
    ("protein-tertiary-structure", "Protein"),
    ("wine-quality-red", "Wine"),
    ("yacht", "Yacht"),
]
UCI_METHOD_ORDER = ["deterministic", "normal", "mixednormal", "sample"]
UCI_BASELINE_METHOD_ORDER = UCI_METHOD_ORDER + ["crps_ensemble", "NDP"]
AR_EXPERIMENT_ORDER = [
    ("Burgers", "1D_Burgers", r"\textbf{Burgers'}"),
    ("KS", "1D_KS", r"\textbf{KS}"),
    ("T2M", "WeatherBench", r"\textbf{T2M}"),
]
AR_METHOD_ORDER = ["deterministic", "normal", "mixednormal", "mvnormal", "sample"]
AR_BASELINE_METHOD_ORDER = AR_METHOD_ORDER + ["iDDPM", "crps_ensemble", "NDP"]

METRIC_SPECS = {
    # metric_key: (column_prefix, header_label, direction, scales_with_factor)
    "rmse":  ("RMSETest",           r"RMSE$\downarrow$",               "min", True),
    "es":    ("EnergyScoreTest",    r"ES$\downarrow$",                 "min", True),
    "crps":  ("CRPSTest",           r"CRPS$\downarrow$",               "min", True),
    "nll":   ("Gaussian NLLTest",   r"NLL$\downarrow$",                "min", False),
    "c95":   ("C95Test",            r"$\mathcal{C}_{0.95}$",           "target:0.95", False),
    "c90":   ("C90Test",            r"$\mathcal{C}_{0.90}$",           "target:0.90", False),
    "c75":   ("C75Test",            r"$\mathcal{C}_{0.75}$",           "target:0.75", False),
    "c50":   ("C50Test",            r"$\mathcal{C}_{0.50}$",           "target:0.50", False),
}

# Per-experiment scaling (applied to RMSE/ES/CRPS only) — matches the legacy tables.
EXPERIMENT_FACTOR = {
    "Burgers": 1000,
    "KS": 100,
    "T2M": 1,
    "UCI": 1,
}

# Per-UCI-dataset scaling overrides (applied to RMSE/ES/CRPS only) for readability.
UCI_DATASET_FACTOR = {
    "naval-propulsion-plant": 10000,
    "kin8nm": 100,
}


def _fmt_mean(x: float) -> str:
    if pd.isna(x):
        return "--"
    return f"{x:.2f}"


def _fmt_std(x: float) -> str:
    # Match example tables: 2dp, trailing zeros stripped (0.50 -> 0.5, 0.57 stays).
    if pd.isna(x):
        return ""
    return format(round(float(x), 2), "g")


def _best_idx(values: list[float], direction: str) -> set[int]:
    finite = [(i, v) for i, v in enumerate(values) if not pd.isna(v)]
    if not finite:
        return set()
    if direction == "min":
        def score(v: float) -> float:
            return v
    elif direction == "max":
        def score(v: float) -> float:
            return -v
    elif direction.startswith("target:"):
        t = float(direction.split(":", 1)[1])

        def score(v: float) -> float:
            return abs(v - t)
    else:
        return set()
    # Ties: everything within rounding (0.005) of the best rounded-to-2dp value wins.
    best_score = min(score(v) for _, v in finite)
    return {i for i, v in finite if abs(score(v) - best_score) < 5e-3}


def _lookup(agg: pd.DataFrame, experiment: str, dataset: str, method: str) -> pd.Series | None:
    sel = agg[(agg.experiment == experiment) & (agg.dataset == dataset) & (agg.method == method)]
    if sel.empty:
        return None
    return sel.iloc[0]


def _cell_makecell(mean: float, std: float, bold: bool, single_run: bool) -> str:
    m = _fmt_mean(mean)
    if bold:
        m = rf"\textbf{{{m}}}"
    if single_run or pd.isna(std):
        return m
    return rf"\makecell{{{m} \\ ($\pm$ {_fmt_std(std)})}}"


def _cell_inline(mean: float, std: float, bold: bool, single_run: bool) -> str:
    m = _fmt_mean(mean)
    s = _fmt_std(std)
    if single_run or not s:
        return rf"\textbf{{{m}}}" if bold else m
    inner = f"{m} ± {s}"
    return rf"\textbf{{{inner}}}" if bold else inner


def build_uci_metric_table(agg: pd.DataFrame, metric_key: str, methods: list[str]) -> str:
    col_prefix, header_label, direction, scales = METRIC_SPECS[metric_key]
    mean_col = f"{col_prefix}_mean"
    std_col = f"{col_prefix}_std"
    n_cols = len(methods)
    col_spec = "l" + "|c" * n_cols
    header_cells = " & ".join(METHOD_SYMBOL.get(m, m) for m in methods)
    lines = [
        rf"\begin{{tabular}}{{{col_spec}}}",
        r"\toprule",
        rf" & {header_cells} \\",
        r"\midrule",
    ]
    for dataset_key, dataset_label in UCI_DATASET_ORDER:
        factor = UCI_DATASET_FACTOR.get(dataset_key, 1) if scales else 1
        row_label = (
            rf"{dataset_label} ($\times {factor}$)"
            if scales and factor != 1
            else dataset_label
        )
        values_mean, values_std, n_runs = [], [], []
        for method in methods:
            if method in {"crps_ensemble", "NDP"}:
                row = _lookup(agg, method, dataset_key, method)
            else:
                row = _lookup(agg, "UCI", dataset_key, method)
            if row is None:
                values_mean.append(float("nan"))
                values_std.append(float("nan"))
                n_runs.append(0)
            else:
                values_mean.append(float(row.get(mean_col, float("nan"))) * factor)
                values_std.append(float(row.get(std_col, float("nan"))) * factor)
                n_runs.append(int(row.get("n_runs", 0)))
        winners = _best_idx(values_mean, direction)
        cells = [
            _cell_inline(values_mean[i], values_std[i], i in winners, n_runs[i] <= 1)
            for i in range(n_cols)
        ]
        lines.append(rf"{row_label} & {' & '.join(cells)} \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(lines) + "\n"


def build_ar_table(agg: pd.DataFrame, metrics: list[str], methods: list[str]) -> str:
    header_cells = " & ".join(METRIC_SPECS[m][1] for m in metrics)
    col_spec = "|l|l|" + "c" * len(metrics) + "|"
    lines = [
        rf" \begin{{tabular}}{{{col_spec}}}",
        r"    \toprule",
        rf"        Experiment & Model & {header_cells} \\",
        r"        \midrule",
    ]
    for idx, (exp, dataset, exp_label) in enumerate(AR_EXPERIMENT_ORDER):
        if idx > 0:
            lines.append(r"        \midrule")
        exp_factor = EXPERIMENT_FACTOR.get(exp, 1)
        # Collect per-method values for this experiment for best-highlighting within experiment.
        per_metric_means: dict[str, list[float]] = {m: [] for m in metrics}
        per_metric_stds: dict[str, list[float]] = {m: [] for m in metrics}
        per_method_n: list[int] = []
        resolved_methods: list[str] = []
        for method in methods:
            if method in {"crps_ensemble", "NDP"}:
                row = _lookup(agg, method, dataset, method)
            else:
                row = _lookup(agg, exp, dataset, method)
            if row is None:
                continue
            resolved_methods.append(method)
            per_method_n.append(int(row.get("n_runs", 0)))
            for mk in metrics:
                prefix, _, _, scales = METRIC_SPECS[mk]
                mult = exp_factor if scales else 1
                per_metric_means[mk].append(float(row.get(f"{prefix}_mean", float("nan"))) * mult)
                per_metric_stds[mk].append(float(row.get(f"{prefix}_std", float("nan"))) * mult)
        # Winners per metric.
        winners_per_metric = {
            mk: _best_idx(per_metric_means[mk], METRIC_SPECS[mk][2])
            for mk in metrics
        }
        # Emit rows.
        n_methods_here = len(resolved_methods)
        lines.append(rf"        \multirow{{{n_methods_here}}}{{*}}{{{exp_label}}}")
        for row_idx, method in enumerate(resolved_methods):
            cells = []
            for mk in metrics:
                cells.append(_cell_makecell(
                    per_metric_means[mk][row_idx],
                    per_metric_stds[mk][row_idx],
                    row_idx in winners_per_metric[mk],
                    per_method_n[row_idx] <= 1,
                ))
            lines.append(rf"        & {METHOD_SYMBOL.get(method, method)} & {' & '.join(cells)} \\")
    lines += [r"        \bottomrule", r"    \end{tabular}"]
    return "\n".join(lines) + "\n"


def write_result_tables(agg: pd.DataFrame, out_dir: str) -> None:
    os.makedirs(out_dir, exist_ok=True)
    # UCI: one table per metric, methods only.
    for key in ["rmse", "es", "crps", "nll", "c95", "c90", "c75", "c50"]:
        tex = build_uci_metric_table(agg, key, UCI_METHOD_ORDER)
        with open(os.path.join(out_dir, f"uci_{key}.tex"), "w") as f:
            f.write(tex)
    # UCI baselines: same per-metric, adds crps_ensemble + NDP.
    for key in ["rmse", "es", "crps", "nll", "c95", "c90", "c75", "c50"]:
        tex = build_uci_metric_table(agg, key, UCI_BASELINE_METHOD_ORDER)
        with open(os.path.join(out_dir, f"uci_baselines_{key}.tex"), "w") as f:
            f.write(tex)
    # AR metrics (RMSE/ES/CRPS/NLL/C95) — methods only.
    tex = build_ar_table(agg, ["rmse", "es", "crps", "nll", "c95"], AR_METHOD_ORDER)
    with open(os.path.join(out_dir, "autoregressive_metrics.tex"), "w") as f:
        f.write(tex)
    # AR coverages — methods only.
    tex = build_ar_table(agg, ["c95", "c90", "c75", "c50"], AR_METHOD_ORDER)
    with open(os.path.join(out_dir, "autoregressive_coverages.tex"), "w") as f:
        f.write(tex)
    # AR metrics with baselines.
    tex = build_ar_table(agg, ["rmse", "es", "crps", "nll", "c95"], AR_BASELINE_METHOD_ORDER)
    with open(os.path.join(out_dir, "autoregressive_baselines_metrics.tex"), "w") as f:
        f.write(tex)
    # AR coverages with baselines.
    tex = build_ar_table(agg, ["c95", "c90", "c75", "c50"], AR_BASELINE_METHOD_ORDER)
    with open(os.path.join(out_dir, "autoregressive_baselines_coverages.tex"), "w") as f:
        f.write(tex)


if __name__ == "__main__":
    results_dir = "results"
    long_df = discover_runs(results_dir)
    long_df.to_csv(f"{results_dir}/all_runs.csv", index=False)
    agg_df = aggregate_runs(long_df)
    agg_df.to_csv(f"{results_dir}/all_aggregated.csv", index=False)
    write_result_tables(agg_df, f"{results_dir}/result_tables")
