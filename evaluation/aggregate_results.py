import os
import glob
import pandas as pd
import numpy as np


def process_experiment(results_dir: str, experiment: str, models: list, factor:int = 1, agg_groups: list = ["distributional_method"]) -> pd.DataFrame:
    metrics = ['t_training_avg', 't_training_std', 'RMSETest','EnergyScoreTest', 'CRPSTest', 'Gaussian NLLTest', 'CoverageTest']
    factor_metrics = ['RMSETest','EnergyScoreTest', 'CRPSTest']
    file_list = []
    for model in models:
        path = f"{results_dir}/{experiment}/{model}/"
        file_list.extend([f for f in glob.iglob(path + "**", recursive = True) if os.path.isfile(f) and f.endswith("test.csv")])
    results = pd.DataFrame()
    for file in file_list:
        if os.path.exists(file):
            results_df = pd.read_csv(file, index_col=0)
            results = pd.concat([results, results_df], axis = 1)
    rows = metrics.copy()
    for g in agg_groups:
        rows.append(g)
    results = results.loc[rows]
    results.loc[metrics] = results.loc[metrics].astype("float32")
    # Multiply by factor
    results.loc[factor_metrics] = results.loc[factor_metrics] * factor
    results = results.transpose()
    # Turn std into var
    results["t_training_std"] = results["t_training_std"]**2
    # Group by uncertainty quantification method
    mean = results.groupby(agg_groups).mean().astype("float32")
    mean.insert(0, "Statistic", "Mean")
    var = results.groupby(agg_groups).var().astype("float32")
    var.insert(0, "Statistic", "Std")
    # Adjust for intra variance
    var["t_training"] = mean["t_training_std"] + var["t_training_avg"]
    var.loc[:,metrics] = np.sqrt(var.loc[:,metrics])
    var.loc[:,"t_training"] = np.sqrt(var.loc[:,"t_training"])
    var.drop(["t_training_avg", "t_training_std"], axis = 1, inplace = True)
    mean = mean.rename({"t_training_avg":"t_training"}, axis = 1).drop("t_training_std", axis = 1)
    results_df = pd.concat([mean.transpose(), var.transpose()], axis = 1)
    results_df = results_df[results_df.columns.sort_values().unique()]
    return results_df

def create_latex_table(results_df: pd.DataFrame, results_dir:str, experiment: str, model: str) -> str:
    # Initialize an empty DataFrame to store the formatted values
    formatted_df = pd.DataFrame()
    metrics = results_df.index[-6:] # Extract only test metrics
    methods = list(results_df.columns.unique())  # Methods are the top level of the columns MultiIndex

    # Create a new DataFrame with the method as the index and metrics as columns
    for metric in metrics:
        formatted_df[metric] = [
            f"\\makecell{{{results_df.loc[metric, method].values[0]:.2f} \\\\ ($\\pm$ {results_df.loc[metric, method].values[1]:.2f})}}"
            for method in methods
        ]
    formatted_df.index = methods
    latex_table = formatted_df.to_latex(escape=False)
    #  save to a file
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
    #  save to a file
    with open(f"{results_dir}/{experiment}/aggregated_results_mean.tex", "w") as f:
        f.write(latex_table)


if __name__ == "__main__":
    experiment = "Burgers"
    models = ["deterministic", "normal", "mixednormal", "mvnormal", "sample"]
    results_dir = "results"
    factor = 1000

    
    results = process_experiment(results_dir, experiment, models, factor = factor)
    results.to_csv(f"{results_dir}/{experiment}/aggregated_results.csv")
    create_latex_table(results, results_dir, experiment, models)