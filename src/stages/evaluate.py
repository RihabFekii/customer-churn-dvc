import json
import logging
import os
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import pandas as pd
import yaml
from box import ConfigBox
from sklearn.metrics import (ConfusionMatrixDisplay, accuracy_score,
                             confusion_matrix, f1_score, roc_auc_score)


logging.basicConfig(level=logging.INFO, format="EVAL: %(message)s")


def evaluate(trained_model_path, processed_data_dir, reports_dir, metrics_fname, cm_fname):
    """Evaluate model."""

    x_test = pd.read_csv(os.path.join(processed_data_dir, "x_test.csv"))
    y_test = pd.read_csv(os.path.join(processed_data_dir, "y_test.csv"))

    model = joblib.load(trained_model_path)
    logging.info("Model loaded!")

    logging.info("Predicting on the test set...")
    y_pred = model.predict(x_test)

    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_pred)

    metrics = {
        'accuracy_score': accuracy,
        'f1_score': f1,
        'roc_auc_score': roc_auc
    }

    metrics_file = Path(reports_dir/metrics_fname)
    # saving metrics in json file
    logging.info(f"Saving metrics to {metrics_file}") 
    with open(metrics_file, 'w') as f:
        f.write(json.dumps(metrics, indent=4))
    
    # confusion matrix 
    cm_file = Path(reports_dir/cm_fname)
    cm = confusion_matrix(y_test, y_pred, normalize='true')
    logging.info(f"Saving confusion matrix to {cm_file}") 
    ConfusionMatrixDisplay(cm).plot()
    plt.savefig(cm_file)


if __name__ == '__main__':

    params_file_path = "params.yaml"
    with open(params_file_path, "r") as f:
        params = yaml.safe_load(f)
        params = ConfigBox(params)

    trained_model_path = Path(params.eval.model_path)
    model_type = params.train.model_type
    logging.info(f"Loading {model_type} trained model...")

    processed_data_dir = Path(params.data_split.processed_data_dir)
    reports_dir = Path(params.eval.reports_dir)
    metrics_fname = Path(params.eval.metrics_fname)
    cm_fname = Path(params.eval.cm_fname)

    evaluate(trained_model_path=trained_model_path,
             processed_data_dir=processed_data_dir,
             reports_dir=reports_dir,
             metrics_fname=metrics_fname,
             cm_fname=cm_fname)



