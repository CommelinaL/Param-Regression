import numpy as np
import pandas as pd
import sys
import os
from dataset import QuaternaryData, FlatData, PROJECT_ROOT
sys.path.append(PROJECT_ROOT, "Param-Regression", "src-py")
from sklearn.neural_network import MLPRegressor
from sklearn.cross_decomposition import PLSRegression
from sklearn.multioutput import MultiOutputRegressor
from sklearn.linear_model import LinearRegression, Lasso, Ridge, BayesianRidge, ElasticNet
from sklearn.svm import SVR
from sklearn.metrics import root_mean_squared_error, r2_score, median_absolute_error
from xgboost import XGBRegressor
#from statsmodels.gam.api import GLMGam, BSplines
#from statsmodels.genmod.families import Gaussian
#from file_processor import process_files
from feat import extract_features_batch
from utils import normalize
import time
import pickle


feat_dict = {'none': [],'npc': list(range(8)), 'rcl': [8, 9, 10], 'cvl': 11, 'san': [12, 13], 'dsa': 14, 'daa': 15, 'cvl_dsa': [11, 14]}
del_feat = 'daa'
model_name = "MLP"
model_dict = {"PLS": PLSRegression(n_components=2),
              "MLP": MLPRegressor(random_state=1, max_iter=5000, early_stopping=True, hidden_layer_sizes=(100, 100, 100), solver='adam'),
              "SVR": MultiOutputRegressor(SVR(kernel='rbf')),
              "Linear": LinearRegression(),
              "Ridge": Ridge(alpha=0.1),
              "XGBoost": XGBRegressor(n_estimators=50, random_state=42, objective='reg:squaredlogerror')}
print(del_feat, model_name)

def evaluate_model(model, X_train, X_test, y_train, y_test, size, save = True):
    save_path = os.path.join('saved_model', 'dataset_size_test', model_name+'_wo_' + del_feat)
    if save and os.path.exists(save_path) == False:
        os.makedirs(save_path)
    file_path = os.path.join(save_path, str(size) +'.pickle')
    print("Training", size)
    start_train = time.time()
    model.fit(X_train, y_train)
    train_time = time.time() - start_train
    print("Testing", size)
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
    return {'dataset size': size, "RMSE (normalized)": rmse_n, 'R2 (normalized)': r2_n, 'MedAE (normalized)': medae_n,
            'RMSE': rmse, 'R2': r2, 'MedAE': medae,
            "train_time": train_time * 1e3, "test_time": test_time * 1e3}

# Data preprocessing
root_dir = os.path.join(PROJECT_ROOT, "pseudo_label", "heuristic_split_dataset")

train_dataset = FlatData(os.path.join(PROJECT_ROOT, "pseudo_label", "train_500k"), "train")
test_dataset = FlatData(os.path.join(root_dir, "test"), "test")
sup_root_dir = os.path.join(PROJECT_ROOT, "pseudo_label", "supplement")
sup_test_dataset = FlatData(os.path.join(sup_root_dir, "test"), "test")

train_point_list = train_dataset.point_list[:250000]
train_target_list = train_dataset.target_list[:250000]
test_point_list = test_dataset.point_list + sup_test_dataset.point_list
test_target_list = test_dataset.target_list + sup_test_dataset.target_list


X_train_orig = extract_features_batch(train_point_list)
X_test = extract_features_batch(test_point_list)
X_train_orig = np.delete(X_train_orig, feat_dict[del_feat], 1)
X_test = np.delete(X_test, feat_dict[del_feat], 1)
y_train_orig = np.array(train_target_list)
y_test = np.array(test_target_list)

results = []
for size in range(10000, 250001, 10000):
    X_train, y_train = X_train_orig[:size], y_train_orig[:size]
    model = model_dict[model_name]
    results.append(evaluate_model(model, X_train, X_test, y_train, y_test, size))
results_df = pd.DataFrame(results)
print(results_df)
results_df.to_csv('dataset_test_'+model_name+'_wo_'+del_feat+'.csv')