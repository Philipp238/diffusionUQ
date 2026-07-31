"""Config-driven Treeffuser baseline on UCI datasets.

Standalone entry point that mirrors ``main.py``'s IO conventions (config
file, timestamped results dir, per-split ``test.csv``) but bypasses the
PyTorch trainer since Treeffuser is a gradient-boosted-tree model
(Beltran-Velez et al., 2024, arXiv:2406.07658). Reuses ``get_uci_data`` so
splits, standardization and ``validation_ratio`` match the neural baselines.

Supports two data regimes:
    * ``validation_ratio > 0``: carve a val split out of train (same as NDP
      configs); Treeffuser fits on the shrunk train, we also log metrics on
      val + test.
    * ``validation_ratio = 0``: no external val; fit on all of train, log
      metrics on test only. Treeffuser's *internal* early-stopping split is
      controlled independently via ``early_stopping_rounds`` / ``eval_percent``
      in the config (set ``early_stopping_rounds = None`` to disable).
"""

import argparse
import ast
import configparser
import datetime
import logging
import os
import pathlib
import shutil
from time import time

import numpy as np
import pandas as pd
import torch
from scoringrules import crps_ensemble, energy_score
from treeffuser import Treeffuser

from data.data_utils import get_uci_data
from utils import losses, train_utils


# Raw-index positions of categorical columns for each UCI dataset. Mirrors
# ``_preprocess_uci_feature_set`` in data/data_utils.py, but we use these
# indices to forward ``cat_idx`` to LightGBM instead of one-hot-expanding.
_CAT_IDX_MAP = {
    "bostonHousing": [3],
    "energy": [4, 6, 7],
    "naval-propulsion-plant": [0, 1, 8, 11],
}


def _to_numpy(t):
    return t.detach().cpu().numpy() if isinstance(t, torch.Tensor) else np.asarray(t)


def _stack_dataset(dataset):
    """Extract full (X, y) numpy arrays from a ``RegressionDataset``.

    The dataset stores standardized tensors in ``.input`` / ``.target`` (or
    the originals when ``standardize=False``); either way we want the
    already-preprocessed values.
    """
    X = _to_numpy(dataset.input)
    y = _to_numpy(dataset.target)
    if X.ndim == 3:
        X = X.reshape(X.shape[0], -1)
    if y.ndim == 3:
        y = y.reshape(y.shape[0], -1)
    if y.ndim == 1:
        y = y.reshape(-1, 1)
    return X, y


def _destandardize(x, std_params, key):
    mean = std_params[key]["mean"].numpy()
    std = std_params[key]["std"].numpy()
    return x * std + mean


def _load_uci_data_native_cat(dataset_name, split, standardize,
                              validation_ratio, data_fraction):
    """Load a UCI split keeping categorical columns as raw integer codes.

    Mirrors the essential parts of ``get_uci_data`` but skips
    ``_preprocess_uci_feature_set``'s one-hot expansion and skips X
    standardization (LightGBM is scale-invariant; also standardizing
    integer-coded categoricals would break their coding). Returns numpy
    arrays plus ``cat_idx`` in the *reordered* column space so LightGBM
    can use native categorical splits.

    Returns:
        X_train, y_train, X_val, y_val, X_test, y_test, std_params, cat_idx
        (X_val / y_val are None when ``validation_ratio == 0``).
    """
    data_directory_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "data", "UCI_Datasets", dataset_name, "data",
    )
    data = np.loadtxt(f"{data_directory_path}/data.txt")
    index_features = np.loadtxt(f"{data_directory_path}/index_features.txt")
    index_target = np.loadtxt(f"{data_directory_path}/index_target.txt")

    feat_cols = [int(i) for i in np.atleast_1d(index_features).tolist()]
    X = data[:, feat_cols]
    y = data[:, [int(np.atleast_1d(index_target).tolist()[0])]]

    cat_idx_raw = _CAT_IDX_MAP.get(dataset_name, [])

    index_train = np.loadtxt(f"{data_directory_path}/index_train_{split}.txt")
    index_test = np.loadtxt(f"{data_directory_path}/index_test_{split}.txt")

    X_tr_full = X[[int(i) for i in index_train.tolist()]]
    y_tr_full = y[[int(i) for i in index_train.tolist()]]
    if data_fraction < 1.0:
        n_sub = max(1, int(data_fraction * X_tr_full.shape[0]))
        X_tr_full = X_tr_full[:n_sub]
        y_tr_full = y_tr_full[:n_sub]

    X_test = X[[int(i) for i in index_test.tolist()]]
    y_test = y[[int(i) for i in index_test.tolist()]]

    if validation_ratio > 0:
        n_train = int((1 - validation_ratio) * X_tr_full.shape[0])
        X_val = X_tr_full[n_train:]
        y_val = y_tr_full[n_train:]
        X_train = X_tr_full[:n_train]
        y_train = y_tr_full[:n_train]
    else:
        X_train = X_tr_full
        y_train = y_tr_full
        X_val = None
        y_val = None

    # Round categorical columns so LightGBM sees clean integer codes.
    for c in cat_idx_raw:
        X_train[:, c] = np.rint(X_train[:, c])
        X_test[:, c] = np.rint(X_test[:, c])
        if X_val is not None:
            X_val[:, c] = np.rint(X_val[:, c])

    if standardize:
        y_mean = y_train.mean(axis=0, keepdims=True)
        y_std = y_train.std(axis=0, keepdims=True)
        y_train_s = (y_train - y_mean) / y_std
        y_test_s = (y_test - y_mean) / y_std
        y_val_s = (y_val - y_mean) / y_std if y_val is not None else None
        std_params = {
            "target": {
                "mean": torch.from_numpy(y_mean.astype(np.float32)),
                "std": torch.from_numpy(y_std.astype(np.float32)),
            },
        }
        y_train, y_test, y_val = y_train_s, y_test_s, y_val_s
    else:
        std_params = None

    return (
        X_train.astype(np.float32), y_train.astype(np.float32),
        None if X_val is None else X_val.astype(np.float32),
        None if y_val is None else y_val.astype(np.float32),
        X_test.astype(np.float32), y_test.astype(np.float32),
        std_params, list(cat_idx_raw),
    )


def _eval_split(name, X, y, model, n_samples, standardize, std_params,
                alpha, results_dict, logger):
    """Sample from Treeffuser and log metrics with the same keys main.py uses."""
    t0 = time()
    # samples: (n_samples, batch, y_dim)
    samples = model.sample(X, n_samples=n_samples, n_parallel=10, n_steps=50)
    t_sample = time() - t0

    y_dim = y.shape[1]

    if standardize and std_params is not None:
        y = _destandardize(y, std_params, "target")
        samples = _destandardize(samples, std_params, "target")

    # Shapes for existing metric utilities:
    #   target_t:     (batch, 1, y_dim)
    #   pred_t:       (batch, 1, y_dim, n_samples)
    target_t = torch.from_numpy(y).float().unsqueeze(1)
    pred_np = np.transpose(samples, (1, 2, 0))  # (batch, y_dim, n_samples)
    pred_t = torch.from_numpy(pred_np).float().unsqueeze(1)

    mse_loss = torch.nn.MSELoss()
    mse = mse_loss(pred_t.mean(dim=-1), target_t).item()

    es = energy_score(
        target_t.flatten(start_dim=1, end_dim=-1),
        pred_t.flatten(start_dim=1, end_dim=-2),
        m_axis=-1, v_axis=-2, backend="torch",
    ).mean().item()

    crps = crps_ensemble(target_t, pred_t, backend="torch").mean().item()

    nll = losses.GaussianNLL()(pred_t, target_t).item()
    cov = losses.Coverage(alpha)(pred_t, target_t, ensemble_dim=-1).item()

    qice_loss = losses.QICE()
    qice_loss.aggregate(pred_t, target_t)
    qice = qice_loss.compute()

    au = pred_t.var(dim=-1).mean().item()

    train_utils.log_and_save_evaluation(mse, "MSE" + name, results_dict, logger)
    train_utils.log_and_save_evaluation(np.sqrt(mse), "RMSE" + name, results_dict, logger)
    train_utils.log_and_save_evaluation(es, "EnergyScore" + name, results_dict, logger)
    train_utils.log_and_save_evaluation(crps, "CRPS" + name, results_dict, logger)
    train_utils.log_and_save_evaluation(nll, "Gaussian NLL" + name, results_dict, logger)
    train_utils.log_and_save_evaluation(cov, "Coverage" + name, results_dict, logger)
    train_utils.log_and_save_evaluation(qice, "QICE" + name, results_dict, logger)
    train_utils.log_and_save_evaluation(au, "AleatoricUncertainty" + name, results_dict, logger)
    # Treeffuser has no analytic second-order head — EU is undefined.
    train_utils.log_and_save_evaluation(float("nan"), "EpistemicUncertainty" + name, results_dict, logger)

    logger.info(f"Sampling on {name} ({X.shape[0]} rows) took {t_sample:.2f}s.")


def run(config_name, results_folder=None):
    config = configparser.ConfigParser()
    config.optionxform = str
    config.read(os.path.join("config", config_name))

    results_path = config["META"]["results_path"]
    experiment_name = config["META"]["experiment_name"]

    d_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_")
    directory = results_folder or os.path.join(results_path, d_time + experiment_name)
    pathlib.Path(directory).mkdir(parents=True, exist_ok=True)
    shutil.copy(os.path.join("config", config_name), directory)
    print(f"Created directory {directory}")

    logging.basicConfig(
        filename=os.path.join(directory, "experiment.log"),
        level=logging.INFO,
        force=True,
    )
    logger = logging.getLogger(__name__)
    logger.info(f"Starting Treeffuser experiment with config {config_name}")

    data_params = {k: ast.literal_eval(v) for k, v in config.items("DATAPARAMETERS")}
    model_params = {k: ast.literal_eval(v) for k, v in config.items("MODELPARAMETERS")}

    dataset_names = data_params["dataset_name"]
    if isinstance(dataset_names, str):
        dataset_names = [dataset_names]
    splits = data_params["yarin_gal_uci_split_indices"]
    if isinstance(splits, int):
        splits = [splits]
    seeds = model_params.get("seed", [1234])
    if isinstance(seeds, int):
        seeds = [seeds]

    standardize = data_params["standardize"]
    validation_ratio = float(data_params["validation_ratio"])
    data_fraction = float(data_params.get("data_fraction", 1.0))
    use_native_cat = bool(data_params.get("use_native_categorical", False))
    alpha = float(model_params.get("alpha", 0.05))
    n_samples_uq = int(model_params["n_samples_uq"])

    # Treeffuser hyperparameters (paper defaults if unspecified).
    tf_kwargs = dict(
        n_repeats=int(model_params.get("n_repeats", 30)),
        n_estimators=int(model_params.get("n_estimators", 3000)),
        early_stopping_rounds=model_params.get("early_stopping_rounds", 50),
        eval_percent=float(model_params.get("eval_percent", 0.1)),
        num_leaves=int(model_params.get("num_leaves", 31)),
        max_depth=int(model_params.get("max_depth", -1)),
        learning_rate=float(model_params.get("learning_rate", 0.1)),
        min_child_samples=int(model_params.get("min_child_samples", 20)),
        subsample=float(model_params.get("subsample", 1.0)),
        subsample_freq=int(model_params.get("subsample_freq", 0)),
        n_jobs=int(model_params.get("n_jobs", -1)),
        sde_name=model_params.get("sde_name", "vesde"),
        sde_initialize_from_data=bool(model_params.get("sde_initialize_from_data", False)),
        verbose=int(model_params.get("verbose", 0)),
    )

    # ``validation_ratio`` in configs is expressed relative to the *full* set;
    # ``get_uci_data`` expects it relative to the train set, matching main.py.
    val_ratio_on_train = (
        validation_ratio / (1 - validation_ratio) if validation_ratio > 0 else 0.0
    )

    # Row-per-split results table (matches aggregator.py expectations).
    results_dict = {
        "dataset_name": [],
        "yarin_gal_uci_split_indices": [],
        "seed": [],
        "data_fraction": [],
        "n_estimators": [],
        "n_repeats": [],
        "sde_name": [],
        "t_training": [],
    }

    for dataset_name in dataset_names:
        for split in splits:
            for seed in seeds:
                logger.info(
                    f"### dataset={dataset_name} split={split} seed={seed} ###"
                )
                np.random.seed(seed)
                torch.manual_seed(seed)

                if use_native_cat:
                    (X_train, y_train, X_val, y_val, X_test, y_test,
                     std_params, cat_idx) = _load_uci_data_native_cat(
                        dataset_name, split, standardize,
                        val_ratio_on_train, data_fraction,
                    )
                else:
                    (datasets, _image_dim, _label_dim) = get_uci_data(
                        dataset_name,
                        splits=split,
                        standardize=standardize,
                        validation_ratio=val_ratio_on_train,
                        data_fraction=data_fraction,
                    )
                    if val_ratio_on_train > 0:
                        train_ds, val_ds, test_ds = datasets
                    else:
                        train_ds, test_ds = datasets
                        val_ds = None
                    X_train, y_train = _stack_dataset(train_ds)
                    X_test, y_test = _stack_dataset(test_ds)
                    X_val, y_val = (_stack_dataset(val_ds) if val_ds is not None
                                    else (None, None))
                    std_params = train_ds.get_std_params() if standardize else None
                    cat_idx = None

                logger.info(
                    f"train={X_train.shape}, test={X_test.shape}"
                    + (f", val={X_val.shape}" if X_val is not None else "")
                    + (f", cat_idx={cat_idx}" if cat_idx else "")
                )

                model = Treeffuser(seed=seed, **tf_kwargs)

                t0 = time()
                model.fit(X_train, y_train, cat_idx=cat_idx if cat_idx else None)
                t_train = time() - t0
                logger.info(f"Fitting Treeffuser took {t_train:.2f}s.")

                results_dict["dataset_name"].append(dataset_name)
                results_dict["yarin_gal_uci_split_indices"].append(int(split))
                results_dict["seed"].append(int(seed))
                results_dict["data_fraction"].append(data_fraction)
                results_dict["n_estimators"].append(tf_kwargs["n_estimators"])
                results_dict["n_repeats"].append(tf_kwargs["n_repeats"])
                results_dict["sde_name"].append(tf_kwargs["sde_name"])
                results_dict["t_training"].append(float(round(t_train, 3)))

                if X_val is not None:
                    _eval_split("Validation", X_val, y_val, model, n_samples_uq,
                                standardize, std_params, alpha, results_dict, logger)
                _eval_split("Test", X_test, y_test, model, n_samples_uq,
                            standardize, std_params, alpha, results_dict, logger)

                # Pad any per-split fields that weren't populated (e.g. Validation
                # metrics when no val set) so the DataFrame stays rectangular.
                n_rows = len(results_dict["seed"])
                for k, v in results_dict.items():
                    while len(v) < n_rows:
                        v.append(float("nan"))

                pd.DataFrame(results_dict).T.to_csv(os.path.join(directory, "test.csv"))

    logger.info("Done.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train and evaluate Treeffuser on UCI datasets.")
    parser.add_argument("-c", "--config", required=True, help="Config file under config/")
    parser.add_argument("-f", "--results_folder", default=None,
                        help="Optional explicit results dir (else auto-timestamped).")
    args = parser.parse_args()
    run(args.config, results_folder=args.results_folder)
