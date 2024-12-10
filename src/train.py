from sys import meta_path
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import pickle
import yaml
from sklearn.metrics import accuracy_score,classification_report,confusion_matrix
from mlflow.models import infer_signature
import os
from sklearn.model_selection import train_test_split,GridSearchCV
from urllib.parse import urlparse  #to get schema of our remote repository enviroment

import mlflow


os.environ['MLFLOW_TRACKING_URI'] = "https://dagshub.com/its-amann/dvc_data_pipeline.mlflow"
os.environ["MLFLOW_TRACKING_USERNAME"] = "its-amann"
os.environ["MLFLOW_TRACKING_PASSWORD"] = "938bac106d07d065d468a6b1338766db5fc4bf55"


def hyperparameter_tuning(X_train,y_train,param_grid):
    rf = RandomForestClassifier()
    grid_search = GridSearchCV(estimator = rf, param_grid = param_grid,cv=3,n_jobs=-1,verbose=2)
    grid_search.fit(X_train, y_train)
    return grid_search

# load parameters from param.yaml
pram = yaml.safe_load(open('params.yaml'))['train']


def train(data_path, model_path, random_state, n_estimators, max_depth):
    data = pd.read_csv(data_path)
    X =  data.drop(columns=['Outcome'])
    y = data['Outcome']


    mlflow.set_tracking_uri("https://dagshub.com/its-amann/dvc_data_pipeline.mlflow")

    with mlflow.start_run():

        Xtrain, Xtest, ytrain, ytest = train_test_split(X, y, test_size=0.2, random_state=random_state)
        signature = infer_signature(Xtrain,ytrain)

        # define parameter grid
        param_grid = {
            'n_estimators': [100,200],
            'max_depth': [5,10,None],
            'min_samples_split': [2,5,10],
            "min_samples_leaf": [1,2,4]
        } 

        # perfom hyperaparameter tuning
        grid_search = hyperparameter_tuning(Xtrain,ytrain,param_grid)

        # get the best model
        best_model = grid_search.best_estimator_

        # predict and evaluate the model
        y_pred = best_model.predict(Xtest)
        accuracy = accuracy_score(ytest,y_pred)
        print(f"Accuracy: {accuracy}")


        # log additional metrics
        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_param("Best_n_estimators", best_model.n_estimators)
        mlflow.log_param("Best_max_depth", best_model.max_depth)
        mlflow.log_param("Best_min_samples_split", best_model.min_samples_split)
        mlflow.log_param("Best_min_samples_leaf", best_model.min_samples_leaf)


        # log confusion matrix and classification report
        confusion = confusion_matrix(ytest,y_pred)
        classification = classification_report(ytest,y_pred)
        mlflow.log_text(str(confusion), "confusion_matrix.txt")
        mlflow.log_text(classification, "classification_report.txt")

        tracking_url_type_store = urlparse(mlflow.get_tracking_uri()).scheme

        if tracking_url_type_store!= 'file':
            mlflow.sklearn.log_model(best_model, "model",registered_model_name="Best Model")
        else : 
            mlflow.sklearn.log_model(best_model, "model", signature=signature)


        # create a dir to save the model
        os.makedirs(os.path.dirname(model_path), exist_ok=True)

        file_name = model_path
        pickle.dump(best_model, open(file_name, 'wb'))


        print(f"Model saved at: {model_path}")

if __name__ == '__main__':
    train(pram['data'],pram['model'],pram['random_state'],pram['n_estimators'],pram['max_depth'])