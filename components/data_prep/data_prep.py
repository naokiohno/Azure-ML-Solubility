
# Load libraries
import os
import argparse
import pandas as pd
from sklearn.model_selection import train_test_split
import logging
import mlflow

# Additional preprocessing steps
from sklearn import preprocessing
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import VarianceThreshold

def main():
    """Main function of the script."""

    # input and output arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, help="path to input data")
    parser.add_argument("--test_train_ratio", type=float, required=False, default=0.25)
    parser.add_argument("--train_data", type=str, help="path to train data")
    parser.add_argument("--test_data", type=str, help="path to test data")
    args = parser.parse_args()

    # Start Logging
    mlflow.start_run()

    print(" ".join(f"{k}={v}" for k, v in vars(args).items()))

    print("input data:", args.data)

    sol_df = pd.read_csv(args.data, header=0, index_col=0)

    mlflow.log_metric("num_samples", sol_df.shape[0])
    mlflow.log_metric("num_features", sol_df.shape[1] - 1)

    sol_train_df, sol_test_df = train_test_split(
        sol_df,
        test_size=args.test_train_ratio,
    )

    # Create preprocessing pipeline
    estimators = [('center_and_scale', preprocessing.StandardScaler()),
                  ('impute', SimpleImputer(strategy='median')),
                  ('remove_zv', VarianceThreshold())]

    solTrainX = sol_train_df.drop(columns=['target'])
    solTrainY = sol_train_df.filter(['target'])

    solTestX = sol_test_df.drop(columns=['target'])
    solTestY = sol_test_df.filter(['target'])

    pipe = Pipeline(estimators)
    pipe.fit(solTrainX)
    train_x_trans = pipe.transform(solTrainX)
    test_x_trans = pipe.transform(solTestX)

    # Rejoin transformed data
    sol_train_df_final = pd.concat([solTrainX, solTrainY], axis=1)
    sol_test_df_final = pd.concat([solTestX, solTestY], axis=1)

    # output paths are mounted as folder, therefore, we are adding a filename to the path
    sol_train_df_final.to_csv(os.path.join(args.train_data, "data.csv"), index=False)
    sol_test_df_final.to_csv(os.path.join(args.test_data, "data.csv"), index=False)

    # Stop Logging
    mlflow.end_run()


if __name__ == "__main__":
    main()
