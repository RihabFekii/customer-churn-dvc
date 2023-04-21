import logging
import os
from pathlib import Path

import joblib
import pandas as pd
import yaml
from box import ConfigBox
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder, StandardScaler
from xgboost import XGBClassifier


logging.basicConfig(level=logging.DEBUG, format='TRAIN: %(message)s')


def train(cat_cols, num_cols, processed_data_dir, model_dir, **train_params):

    x_train = pd.read_csv(os.path.join(processed_data_dir, "x_train.csv"))
    y_train = pd.read_csv(os.path.join(processed_data_dir, "y_train.csv"))

    # Create Pipelines for Numerical Features
    numerical_pipeline = Pipeline(
        steps=[
        ("imputer", SimpleImputer(strategy='mean')),
        ("scaler", StandardScaler())
        ]
    )
    # Create Pipelines for Categorical features
    categorical_pipeline = Pipeline(
        steps=[
        ("imputer", SimpleImputer(strategy='most_frequent')),
        ("encoder", OrdinalEncoder())
        ]
    )

    # Combine the Numerical and Categorical Features preporcessing pipelines with a Column transformer 
    preprocessor = ColumnTransformer(
        transformers=[
        ("num_transformer", numerical_pipeline, num_cols),
        ("cat_transpormer", categorical_pipeline, cat_cols)
        ]
    )

    # Classification model
    clf = XGBClassifier(**train_params)

    # Training pipeline 
    model = Pipeline(
        steps=[
        ("preprocessor", preprocessor),
        ("classifier", clf)
        ]
    )

    # Train model
    logging.info("Starting training") 
    trained_model = model.fit(x_train, y_train)
    
    # Save trained model 
    logging.info("Saving model")
    joblib.dump(trained_model, filename=model_dir/"model.pkl")


if __name__ == "__main__":

    params_file_path = "params.yaml"
    with open(params_file_path, "r") as f:
        params = yaml.safe_load(f)
        params = ConfigBox(params)

    train_params = params.train.train_params
    model_type = params.train.model_type
    logging.info(f"{model_type} training params: {train_params}")
    processed_data_dir = Path(params.data_split.processed_data_dir)
    model_dir = Path(params.train.model_dir)

    logging.info("Starting to train...")
    train(cat_cols=params.data.cat_cols, 
          num_cols=params.data.num_cols,
          processed_data_dir=processed_data_dir, 
          model_dir=model_dir,
          **train_params
          )
    print("Done!")
    