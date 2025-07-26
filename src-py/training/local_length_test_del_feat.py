import numpy as np
import pandas as pd
import sys
sys.path.append(r"D:\BSplineLearning\Param-Regression\src-py")
from sklearn.linear_model import LinearRegression, Lasso, Ridge, BayesianRidge, ElasticNet
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, AdaBoostRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.cross_decomposition import PLSRegression
from sklearn.gaussian_process import GaussianProcessRegressor, kernels
from sklearn.gaussian_process.kernels import RBF
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import root_mean_squared_error, r2_score, median_absolute_error
#from statsmodels.gam.api import GLMGam, BSplines
#from statsmodels.genmod.families import Gaussian
from sklearn.multioutput import MultiOutputRegressor
from xgboost import XGBRegressor
from file_processor import process_files
from dataset import QuaternaryData, FlatData
from feat import extract_features_batch
from utils import normalize
import os
import time
import pickle
from custom_models import MyPolyRegressor, MyPCR, MultiLinearGAM, IsoMapRegressor, LLERegressor, SpectralRegressor



class rbf_normalized(RBF):
    def __call__(self, X, Y=None, eval_gradient=False):
        if Y is not None:
            if eval_gradient == True:
                raise ValueError("Gradient can only be evaluated when Y is None.")
            return super().__call__(normalize(X), normalize(Y))
        else:
            if eval_gradient == False:
                return super().__call__(normalize(X))
            #X_n, grad_n = normalize(X, eval_gradient=True)
            X_n = normalize(X)
            K, grad_k = super().__call__(X_n, eval_gradient=True)
            #return K, np.expand_dims(np.tensordot(np.squeeze(grad_k), grad_n, [-1, 0]), -1)
            return K, grad_k

# Data preprocessing
local_len = 15
del_feat = 'npc'
print(local_len, del_feat)
feat_dict = {'none': [], 'npc': list(range(2*local_len)), 'rcl': list(range(2*local_len, 3*local_len-1)),
             'cvl': 3*local_len-1, 'san': list(range(3*local_len, 4*local_len-2)), 
             'daa': list(range(4*local_len-2, 5*local_len-5)), 'dsa':list(range(5*local_len-5, 6*local_len-8))}
root_dir = r"D:\BSplineLearning\variable_length\split_dataset_" + str(local_len)

# train_dataset = FlatData(os.path.join(root_dir, "train"), "train")
test_dataset = FlatData(os.path.join(root_dir, "test"), "test")
train_dataset = FlatData(r"D:\BSplineLearning\variable_length\train_500k_"+str(local_len), "train")

train_point_list = train_dataset.point_list
train_target_list = train_dataset.target_list
test_point_list = test_dataset.point_list
test_target_list = test_dataset.target_list
train_point_list = train_point_list[:250000]
train_target_list = train_target_list[:250000]

print("train: ", len(train_target_list))
print("test: ", len(test_target_list))

X_train = extract_features_batch(train_point_list, local_len)
X_test = extract_features_batch(test_point_list, local_len)
X_train = np.delete(X_train, feat_dict[del_feat], 1)
X_test = np.delete(X_test, feat_dict[del_feat], 1)
y_train = np.array(train_target_list)
y_test = np.array(test_target_list)

def evaluate_model(model, X_train, X_test, y_train, y_test, model_name, save = True):
    save_path = os.path.join('saved_model', 'seq_'+str(local_len)+"_wo_"+del_feat, str(len(train_target_list)))
    if save and os.path.exists(save_path) == False:
        os.makedirs(save_path)
    file_path = os.path.join(save_path, model_name +'.pickle')
    print("Training", model_name)
    start_train = time.time()
    model.fit(X_train, y_train)
    train_time = time.time() - start_train
    print("Testing", model_name)
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
    return {'Model': model_name, 'RMSE': rmse, 'R2': r2, 'MedAE': medae,
            "RMSE (normalized)": rmse_n, 'R2 (normalized)': r2_n, 'MedAE (normalized)': medae_n,
            "train_time": train_time * 1e3, "test_time": test_time * 1e3}

results = []

# 1. Linear Regression
lr = LinearRegression()
results.append(evaluate_model(lr, X_train, X_test, y_train, y_test, 'Linear Regression'))

# 2. Lasso Regression
lasso = Lasso(alpha=0.1)
results.append(evaluate_model(lasso, X_train, X_test, y_train, y_test, 'Lasso Regression'))

# 3. Ridge Regression
ridge = Ridge(alpha=0.1)
results.append(evaluate_model(ridge, X_train, X_test, y_train, y_test, 'Ridge Regression'))

# 4. Polynomial Regression

results.append(evaluate_model(MyPolyRegressor(degree=2), X_train, X_test, y_train, y_test, 'Quadratic Regression'))
# results.append(evaluate_model(MyPolyRegressor(degree=3), X_train, X_test, y_train, y_test, 'Cubic Regression'))

# 5. Support Vector Regression
# kernel: ['linear', 'poly', 'rbf', 'sigmoid', 'precomputed']
svr = MultiOutputRegressor(SVR(kernel='rbf'))
results.append(evaluate_model(svr, X_train, X_test, y_train, y_test, 'SVR'))

# 6. Decision Trees
dt = DecisionTreeRegressor(random_state=42, min_samples_split=5, min_samples_leaf=2, max_depth=20, criterion='squared_error')
results.append(evaluate_model(dt, X_train, X_test, y_train, y_test, 'Decision Tree'))

# 7. Random Forests
# criterion: {“squared_error”, “absolute_error”, “friedman_mse”, “poisson”}, default=”squared_error”
# rf = RandomForestRegressor(n_estimators=100, random_state=42, criterion='poisson', min_samples_leaf=2, min_samples_split=2)
# results.append(evaluate_model(rf, X_train, X_test, y_train, y_test, 'Random Forest'))

# 8. Gradient Boosted Trees
# criterion: {‘friedman_mse’, ‘squared_error’}, default=’friedman_mse’
# loss: {‘squared_error’, ‘absolute_error’, ‘huber’, ‘quantile’}, default=’squared_error’
# gb = MultiOutputRegressor(GradientBoostingRegressor(n_estimators=100, random_state=42, criterion='friedman_mse', loss='huber', learning_rate=0.1, max_depth=11))
# results.append(evaluate_model(gb, X_train, X_test, y_train, y_test, 'Gradient Boosting'))

# 9. AdaBoost
# loss: {‘linear’, ‘square’, ‘exponential’}, default=’linear’
# ada = MultiOutputRegressor(AdaBoostRegressor(n_estimators=25, random_state=42, loss='linear'))
# results.append(evaluate_model(ada, X_train, X_test, y_train, y_test, 'AdaBoost'))

# 10. XGBoost
# eval_metric: rmse, rmsle, mae, mape, mphe, logloss; default=None
# grow_policy: depthwise, lossguide
# objective: reg:squarederror, reg:squaredlogerror, reg:logistic, reg:pseudohubererror, reg:absoluteerror, reg:quantileerror
xgb = XGBRegressor(n_estimators=50, random_state=42, objective='reg:squaredlogerror')
results.append(evaluate_model(xgb, X_train, X_test, y_train, y_test, 'XGBoost'))

# 11. Principal Component Regression
# results.append(evaluate_model(MyPCR(n_components=4), X_train, X_test, y_train, y_test, 'PCR'))

# 12. Partial Least Squares
pls = PLSRegression(n_components=2)
results.append(evaluate_model(pls, X_train, X_test, y_train, y_test, 'PLS'))

# 13. Bayesian Regression
br = MultiOutputRegressor(BayesianRidge())
results.append(evaluate_model(br, X_train, X_test, y_train, y_test, 'Bayesian Regression'))

# 14. Gaussian Processes
# gp = MultiOutputRegressor(GaussianProcessRegressor(kernel=kernels.Sum(kernels.Matern(), kernels.RationalQuadratic()), random_state=42, copy_X_train=False))
# results.append(evaluate_model(gp, X_train, X_test, y_train, y_test, 'Gaussian Process'))

# 15. Generalized Additive Models
# callbacks: deviance, diffs, coef
gam = MultiLinearGAM(dim = local_len - 1)
results.append(evaluate_model(gam, X_train, X_test, y_train, y_test, 'GAM'))

# 16. Multi-Layer Network
# activation: {‘identity’, ‘logistic’, ‘tanh’, ‘relu’}, default=’relu’
# solver: {‘lbfgs’, ‘sgd’, ‘adam’}, default=’adam’
mlp_feat = MLPRegressor(random_state=1, max_iter=5000, early_stopping=True, hidden_layer_sizes=(100, 100, 100), solver='adam')
results.append(evaluate_model(mlp_feat, X_train, X_test, y_train, y_test, 'MLP with manual feature'))

# mlp_raw = MLPRegressor(random_state=1, max_iter=5000, early_stopping=True, hidden_layer_sizes=(100, 100, 200, 200, 100, 100), solver='adam')
# results.append(evaluate_model(mlp_raw, X_train[:, :8], X_test[:, :8], y_train, y_test, 'MLP with raw data'))

# 17. ElasticNet
elastic = ElasticNet(alpha=0.1)
results.append(evaluate_model(elastic, X_train, X_test, y_train, y_test, 'ElasticNet'))

# 18. Isomap Regression
# isomap_model = IsoMapRegressor(n_components=5)
# results.append(evaluate_model(isomap_model, X_train, X_test, y_train, y_test, 'Isomap'))

# 19. Locally Linear Embedding (LLE)
# lle = LLERegressor(n_components=4, n_neighbors=3)
# results.append(evaluate_model(lle, X_train, X_test, y_train, y_test, 'LLE'))

# 20. Spectral Embedding
# spectral = SpectralRegressor(n_components=5)
# results.append(evaluate_model(spectral, X_train, X_test, y_train, y_test, 'Spectral'))


# Convert results to DataFrame and sort by RMSE
results_df = pd.DataFrame(results)
results_df = results_df.sort_values('RMSE (normalized)')

print(results_df)
results_df.to_csv('results_'+str(local_len)+'_wo_'+del_feat+'_'+str(len(train_target_list))+'.csv')
