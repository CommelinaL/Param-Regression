import numpy as np
import pandas as pd
import sys
sys.path.append(r"D:\BSplineLearning\Param-Regression\src-py")
sys.path.append("/home/lsy/Param-Regression/src-py")
from sklearn.neural_network import MLPRegressor
from sklearn.cross_decomposition import PLSRegression
from sklearn.multioutput import MultiOutputRegressor
from sklearn.linear_model import LinearRegression, Lasso, Ridge, BayesianRidge, ElasticNet
from sklearn.ensemble import RandomForestRegressor
from custom_models import MultiAdaBoost, MultiGBDT, MultiSVR, LLERegressor
from sklearn.svm import SVR
from sklearn.metrics import root_mean_squared_error, r2_score, median_absolute_error
from xgboost import XGBRegressor
#from statsmodels.gam.api import GLMGam, BSplines
#from statsmodels.genmod.families import Gaussian
#from file_processor import process_files
from dataset import QuaternaryData, FlatData
from feat import extract_features_batch
from utils import normalize
import os
import time
import pickle


feat_dict = {'none': [],'npc': list(range(8)), 'rcl': [8, 9, 10], 'cvl': 11, 'san': [12, 13], 'dsa': 14, 'daa': 15, 'cvl_dsa': [11, 14]}

model_name = "PLS"
param_name = "n_components"
param_range = list(range(2,10))
model_dict = {"PLS": (PLSRegression, {"n_components" : 4}),
              "MLP": (MLPRegressor, {"random_state": 1, "max_iter": 5000, "early_stopping": True,
                                     "hidden_layer_sizes":(100, 100), "solver":'adam'}),
              "SVR": (MultiSVR, {"kernel": 'rbf'}),
              "XGBoost": (XGBRegressor, {"random_state": 42, "objective":'reg:squaredlogerror', 'learning_rate': 0.1,
                                         'max_depth': 8, 'colsample_bytree': 1, 'subsample': 0.9}),
              'Random Forest': (RandomForestRegressor, {"random_state": 42, 'max_depth': None}),
              "AdaBoost": (MultiAdaBoost, {"random_state": 42}),
              "Gradient Boosting": (MultiGBDT, {"random_state": 42, 'criterion': 'friedman_mse', 'learning_rate': 0.1, 'loss': 'huber', 'max_depth': 11}),
              "LLE": (LLERegressor, {"n_neighbors": 3, "n_components": 4}),
              "Lasso": (Lasso, {"alpha": 0.1}),
              "Ridge": (Ridge, {"alpha": 0.1}),
              "ElasticNet": (ElasticNet, {"alpha": 0.1}),
}
print(model_name, param_name)

def evaluate_model(model, X_train, X_test, y_train, y_test, test_param, save = True):
    save_path = os.path.join('saved_model', 'param_grid_test', model_name+'_on_' + param_name)
    if save and os.path.exists(save_path) == False:
        os.makedirs(save_path)
    file_path = os.path.join(save_path, str(test_param) +'.pickle')
    print("Training", test_param)
    start_train = time.time()
    model.fit(X_train, y_train)
    train_time = time.time() - start_train
    print("Testing", test_param)
    start_test = time.time()
    y_pred = model.predict(X_test)
    test_time = time.time() - start_test
    
    rmse = root_mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    medae = median_absolute_error(y_test, y_pred)
    y_pred_n = normalize(y_pred)
    rmse_n = root_mean_squared_error(y_test, y_pred_n)
    r2_n = r2_score(y_test, y_pred_n)
    medae_n = median_absolute_error(y_test, y_pred_n)
    with open(file_path, 'wb') as f:
        pickle.dump(model, f)
    return {param_name: test_param, "RMSE (normalized)": rmse_n, 'R2 (normalized)': r2_n, 'MedAE (normalized)': medae_n,
            'RMSE': rmse, 'R2': r2, 'MedAE': medae,
            "train_time": train_time * 1e3, "test_time": test_time * 1e3}

# Data preprocessing
root_dir = "/home/lsy/pseudo_label/heuristic_split_dataset (2)"

train_dataset = FlatData("/home/lsy/pseudo_label/sup_100k", "train")
test_dataset = FlatData(os.path.join(root_dir, "test"), "test")
sup_root_dir = "/home/lsy/pseudo_label/supplement"
sup_test_dataset = FlatData(os.path.join(sup_root_dir, "test"), "test")
# sup2_dataset = FlatData(os.path.join(sup_root_dir, "train"), "train")
#sup3_dataset = FlatData(r"D:\BSplineLearning\pseudo_label\sup_100k", "train")

train_point_list = train_dataset.point_list
train_target_list = train_dataset.target_list
# train_point_list = train_dataset.point_list + sup2_dataset.point_list
# train_target_list = train_dataset.target_list + sup2_dataset.target_list
test_point_list = test_dataset.point_list + sup_test_dataset.point_list
test_target_list = test_dataset.target_list + sup_test_dataset.target_list


X_train = extract_features_batch(train_point_list)
X_test = extract_features_batch(test_point_list)
y_train = np.array(train_target_list)
y_test = np.array(test_target_list)

results = []
model_cls, model_params = model_dict[model_name]
for test_param in param_range:
    model_params[param_name] = test_param
    model = model_cls(**model_params)
    results.append(evaluate_model(model, X_train, X_test, y_train, y_test, test_param))
results_df = pd.DataFrame(results)
print(results_df)
results_df.to_csv('param_test_'+model_name+'_on_'+param_name+'.csv')