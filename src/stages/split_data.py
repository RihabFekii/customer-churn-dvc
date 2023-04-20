import os
from pathlib import Path

import pandas as pd
import yaml
from box import ConfigBox
from sklearn.model_selection import train_test_split


def data_split(
        raw_data_dir,
        processed_data_dir,
        data_file_name,
        target_col,
        cat_cols,
        num_cols,
        random_state,
        test_size):
    ''' 
    Split dataset to train and test sets and save them.
    '''

    data = pd.read_csv(os.path.join(raw_data_dir, data_file_name))
    X = data[cat_cols + num_cols]
    y = data[target_col]
    x_train, x_test, y_train, y_test = train_test_split(X, y, 
                                                        test_size = test_size,
                                                        random_state=random_state,
                                                        stratify=y)
    x_train.to_csv(processed_data_dir/"x_train.csv", index=False)
    x_test.to_csv(processed_data_dir/"x_test.csv", index=False)
    y_train.to_csv(processed_data_dir/"y_train.csv", index=False)
    y_test.to_csv(processed_data_dir/"y_test.csv", index=False)

if __name__ == "__main__":

    params_file_path = "./params.yaml"
    with open(params_file_path, "r") as f:
        params = yaml.safe_load(f)
        params = ConfigBox(params)

    raw_data_dir = Path(params.data.raw_data_dir)
    processed_data_dir = Path(params.data_split.processed_data_dir)
    data_file_name = params.data.data_file_name
    target_col = params.data.target_col
    cat_cols = params.data.cat_cols
    num_cols = params.data.num_cols
    random_state = params.base.random_state
    test_size = params.data_split.test_size

    data_split(
        raw_data_dir=raw_data_dir,
        processed_data_dir=processed_data_dir,
        data_file_name=data_file_name,
        target_col=target_col,
        cat_cols=cat_cols,
        num_cols=num_cols,
        random_state=random_state,
        test_size=test_size)



    
