import mlp
import pandas
import numpy as np
import datetime

from sklearn import *
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error

## Load ettings from config.yaml

ml_config_dict = mlp.setup.load_config()['ml_config']
verbosity = mlp.setup.load_config()['verbosity']

## Some helper functions

def make_XY(df):
    X = df.drop(columns=['total_scooter'])
    Y = df['total_scooter']
    
    return X, Y

def sklearn_create(input_string, kwarg_dict):
    """
    Given an input string, instantiates object from sklearn api library
    """
    if kwarg_dict:
        return eval(f"{input_string}")(**kwarg_dict)
    else:
        return eval(f"{input_string}")()

class Pipeline(object):
    """
    Pipeline class that implements the end-to-end extraction, preprocessing, fitting and evaluation tasks
    """    
    def __init__(self):
        print("Initializing ML pipeline...")
        
        ## Initialize models
        self.reg = sklearn_create(ml_config_dict['chosen_regressor'], ml_config_dict['regressor_params'])
        self.scaler = sklearn_create(ml_config_dict['chosen_scaler'], ml_config_dict['scaler_params'])
        
        ## Extract, preprocess and split data
        self.train_df, self.test_df = mlp.preprocessing.extract_preprocess_split()
        self.X_train, self.Y_train = make_XY(self.train_df)
        self.X_train_scaled = self.scaler.fit_transform(self.X_train)
        
        ## Add params to init_dict dictionary for logging purposes
        self.init_dict = {}
        self.init_dict['datetime'] = str(datetime.datetime.now())
        self.init_dict['model'] = ml_config_dict['chosen_regressor']
        self.init_dict['model_params_user'] = ml_config_dict['regressor_params']
        self.init_dict['scaler'] = ml_config_dict['chosen_scaler']
        self.init_dict['scaler_params_user'] = ml_config_dict['scaler_params']
        
        ## Initialize test_dict logging purposes
        self.test_dict = {}
        
        return

 
    def grid_search(self):
        self.grid_search = GridSearchCV(self.reg, ml_config_dict['param_grid'],
                                       cv=ml_config_dict['cv_folds'],
                                       scoring='neg_mean_squared_error',
                                       return_train_score=True,
                                       verbose=verbosity,
                                       refit=True)

        print("Starting grid search...")
        self.grid_search.fit(self.X_train_scaled, self.Y_train)
        print("Grid search complete")
        self.cv_results = self.grid_search.cv_results_
        self.final_model = self.grid_search.best_estimator_
        
        best_index = self.grid_search.best_index_
        
        # Metrics to extract from cv_results
        eval_metrics = ['mean_train_score', 'mean_test_score' , 'params']

        self.best_metrics = {metric:score[best_index] for metric,score in self.cv_results.items() if metric in eval_metrics}
        self.best_metrics["rmse_train"] = np.sqrt(-self.best_metrics["mean_train_score"])
        self.best_metrics["rmse_cv"] = np.sqrt(-self.best_metrics["mean_test_score"])

        del self.best_metrics['mean_train_score']
        del self.best_metrics['mean_test_score']
        self.best_metrics["model_params_gridsearch"] = self.best_metrics.pop("params")
        return
    
    
    def eval_test(self):
        print("Evaluating on test set...")
        X_test, Y_test = make_XY(self.test_df)
        X_test_scaled=self.scaler.transform(X_test)
        
        Y_pred_test = self.final_model.predict(X_test_scaled)
        self.test_dict["rmse_test"] = np.sqrt(mean_squared_error(Y_test, Y_pred_test))
    
    def get_combined_dict(self):
        return {**self.init_dict, **self.best_metrics, **self.test_dict}