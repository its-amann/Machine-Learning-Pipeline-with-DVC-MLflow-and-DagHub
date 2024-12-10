import pandas as pd
import pickle
from sklearn.metrics import accuracy_score
import yaml
import os
import mlflow
from urllib.parse import urlparse

import mlflow

os.environ['MLFLOW_TRACKING_URI'] = "https://dagshub.com/its-amann/dvc_data_pipeline.mlflow"
os.environ["MLFLOW_TRACKING_USERNAME"] = "its-amann"
os.environ["MLFLOW_TRACKING_PASSWORD"] = "938bac106d07d065d468a6b1338766db5fc4bf55"

params = yaml.safe_load(open('params.yaml'))['evaluate']


def evaluate(data_path,model_path):
    data = pd.read_csv(data_path)
    X = data.drop(columns=['Outcome'])
    y =  data['Outcome']

    mlflow.set_tracking_uri('https://dagshub.com/its-amann/dvc_data_pipeline.mlflow')

    # load the model from the disk

    model = pickle.load(open(model_path, 'rb'))

    predictions = model.predict(X)

    accuracy = accuracy_score(y,predictions)

    # log the metrics to mlflow
    print(f"model Accuracy : {accuracy}")


if __name__ == '__main__':
    evaluate(params['data'],params['model'])
