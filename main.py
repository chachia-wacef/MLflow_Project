import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
import pandas as pd
import os
import fire
from dotenv import load_dotenv
from sklearn.model_selection import GridSearchCV
import logging

load_dotenv()



def main():
    #My remote mlflow instance
    mlflow.set_tracking_uri(os.getenv('MY_MLflow_instance') + ":5000")
    #Create an experiment if it doesn't exist
    mlflow.set_experiment(os.getenv('MY_MLflow_experiment')) 
    #Load data
    iris = pd.read_csv('./Datasets/iris_data.csv')
    x = iris.data[:, 2:]
    y = iris.target
    with mlflow.start_run(run_name=os.getenv('MLflow_run_name')) as run:
        # add parameters for tuning
        num_estimators = 100
        random_state = 42
        mlflow.log_param("num_estimators",num_estimators)
        mlflow.log_param("random_state",random_state)
        
        # train the model
        rf = RandomForestRegressor(n_estimators=num_estimators,random_state=random_state)
        
        #Grid search
        param_grid = {
            'rf_max_depth' : [4,6,8,10],
            'rf_max_features' : [0.5,0.6,0.7,0.8,0.9],
            'rf_max_leaf_nodes' : [5,10,15,20,25]     
        }
        
        # Training
        logging.info("beginning training")
        search = GridSearchCV(rf, param_grid, iid=False, cv=3, return_train_score=False)
        
        search.fit(x, y)
        print(f"Best parameter (CV score={search.best_score_}):")
        print(search.best_params_)
        
        #Log best parameters
        mlflow.log_params(search.best_params_)
        #Log model performance 
        mlflow.log_metric("best_score", search.best_score_)
        
        # save the model artifact for deployment
        logging.info("saving best model")
        mlflow.sklearn.log_model(search.best_estimator_, "best-random-forest-model")

        run_id = run.info.run_uuid
        experiment_id = run.info.experiment_id
        mlflow.end_run()
        print(mlflow.get_artifact_uri())
        print("runID: %s" % run_id)

if __name__ == "__main__":
    logging.basicConfig(format='%(asctime)s - %(message)s', level=logging.INFO)
    fire.Fire(main)