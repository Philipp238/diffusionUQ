import os

import pandas as pd

def csv_aggregator(name:str, input_files: [str], output_files: [str], split_param: str = 'yarin_gal_uci_split_indices'):
    dfs = []
    for i, input_file in enumerate(input_files):
    
        # Load the CSV into a pandas DataFrame
        df = pd.read_csv(input_file, index_col=0)
        df = df.transpose()

        param_cols = ["backbone", "x_T_sampling_method", "distributional_method"]
        eval_cols = ["RMSETest", "EnergyScoreTest", "CRPSTest", "Gaussian NLLTest", "CoverageTest", "QICETest"]

        df[eval_cols] = df[eval_cols].astype(float)

        # Check that the split_param exists
        if split_param not in df.columns:
            raise ValueError(f"The parameter '{split_param}' was not found in the CSV columns.")

        rounding_factor = 8 if (name == "naval" or name == "kin8nm") else 3

        dfs.append(df)

    df = pd.concat(dfs)
    output = df.groupby(param_cols)[eval_cols].mean().round(rounding_factor).reset_index()

    # Save the averaged result to a new CSV
    for output_file in output_files:
        output.to_csv(output_file, index=False)
        print(f"Averaged results saved to {output_file}")


def aggregate():
    datasets = ["energy", "concrete", "kin8nm", "naval", "power", "protein", "wine", "yacht"]
    datasets = ["concrete"]
    dirname = os.path.dirname(__file__)

    for name in datasets:
        file_name = f"results_{name}_CARD_pretrain_CARD_backbone"
        path = os.path.join(dirname, "..", file_name)
        print(f"Path: {path}")
        subdirs = next(os.walk(path))[1]

        input_files = []
        output_files = []
        for subdir in subdirs:
            input_files.append(os.path.join(path, subdir, "test.csv"))
            output_files.append(os.path.join(path, subdir, "agg_test.csv"))

        csv_aggregator(name, input_files, output_files)


if __name__ == "__main__":
    aggregate()