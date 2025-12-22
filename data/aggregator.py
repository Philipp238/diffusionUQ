import os

import pandas as pd

def csv_aggregator(name:str, input_files: [str], output_files: [str], split_param: str = 'yarin_gal_uci_split_indices'):
    dfs = []
    for i, input_file in enumerate(input_files):
    
        # Load the CSV into a pandas DataFrame
        df = pd.read_csv(input_file, index_col=0)
        df = df.transpose()

        param_cols = ["loss_lambda", "n_epochs", "distributional_method", "n_components"]
        eval_cols = ["RMSETest", "EnergyScoreTest", "CRPSTest", "Gaussian NLLTest", "CoverageTest", "QICETest"]

        df[eval_cols] = df[eval_cols].astype(float)

        # Check that the split_param exists
        if split_param not in df.columns:
            raise ValueError(f"The parameter '{split_param}' was not found in the CSV columns.")

        mult = 100 if name == "kin8nm" else 1
        mult = 10000 if name == "naval" else mult

        rounding_factor = 2

        dfs.append(df)

    df = pd.concat(dfs)
    # output = df.groupby(param_cols)[eval_cols].mean().round(rounding_factor).reset_index()

    # # Save the averaged result to a new CSV
    # for output_file in output_files:
    #     output.to_csv(output_file, index=False)
    #     print(f"Averaged results saved to {output_file}")

    grouped = df.groupby(param_cols)[eval_cols].agg(['mean', 'std']).reset_index()

    # Flatten column MultiIndex
    grouped.columns = ['_'.join(filter(None, col)).strip('_') for col in grouped.columns]

    # Format mean ± std for each metric
    for col in eval_cols:
        mean_col = f"{col}_mean"
        std_col = f"{col}_std"

        grouped[col] = (
            (grouped[mean_col] * mult).round(rounding_factor).astype(str) +
            " ± " +
            (grouped[std_col] * mult).round(rounding_factor).astype(str)
        )

        # Optionally drop raw mean/std columns
        grouped.drop(columns=[mean_col, std_col], inplace=True)

    # Save results
    for output_file in output_files:
        grouped.to_csv(output_file, index=False)
        print(f"Averaged results ({grouped.shape[0]} rows) saved to {output_file}")


def aggregate():
    datasets = ["energy", "concrete", "kin8nm", "naval", "power", "protein", "wine", "yacht"]
    datasets = ["energy", "concrete", "kin8nm", "wine", "yacht"]
    dirname = os.path.dirname(__file__)

    for name in datasets:
        file_name = f"results_{name}_CARD_pretrain_CARD_backbone"
        file_name = f"results_{name}_CARD_sampling_and_epochs_likeCARD"
        # file_name = f"results_{name}_experiments"
        # file_name = f"results_{name}_CARD_sampling_and_epochs_likeCARD_2000epochs_5n_comp"
        #file_name = f"results/UCI/{name}"
        file_name = f"results_iDDPM/{name}/"
        path = os.path.join(dirname, "..", file_name)
        print(f"Path: {path}")
        subdirs = next(os.walk(path))[1]

        input_files = []
        output_files = []
        for subdir in subdirs:
            input_files.append(os.path.join(path, subdir, "test.csv"))
            output_files.append(os.path.join(path, subdir, "agg_test_std.csv"))

        csv_aggregator(name, input_files, output_files)


if __name__ == "__main__":
    aggregate()
