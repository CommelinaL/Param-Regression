from sklearn.base import BaseEstimator
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.decomposition import PCA, KernelPCA
from sklearn.utils.optimize import _check_optimize_result
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, AdaBoostRegressor
import scipy
from sklearn.manifold import Isomap, LocallyLinearEmbedding, SpectralEmbedding
from pygam import LinearGAM
import numpy as np

class MyPolyRegressor(BaseEstimator):
    def __init__(self, degree=2):
        self.poly = PolynomialFeatures(degree=degree)
        self.regressor = LinearRegression()
        self.degree = degree
    
    def fit(self, X, y):
        X_poly = self.poly.fit_transform(X)
        self.regressor.fit(X_poly, y)
    
    def predict(self, X):
        X_poly = self.poly.transform(X)
        return self.regressor.predict(X_poly)

# kernel: {‘linear’, ‘poly’, ‘rbf’, ‘sigmoid’, ‘cosine’, ‘precomputed’} or callable, default=’linear’
class MyPCR(BaseEstimator):
    def __init__(self, n_components=5):
        self.pca = KernelPCA(n_components=n_components, kernel='cosine')
        self.regressor = LinearRegression()
    
    def fit(self, X, y):
        X_pca = self.pca.fit_transform(X)
        self.regressor.fit(X_pca, y)
    
    def predict(self, X):
        X_pca = self.pca.transform(X)
        return self.regressor.predict(X_pca)

class MyGPRegressor(GaussianProcessRegressor):
    def __init__(self, 
            kernel=None,
            *,
            alpha=1e-10,
            optimizer="fmin_l_bfgs_b",
            n_restarts_optimizer=0,
            normalize_y=False,
            copy_X_train=True,
            n_targets=None,
            random_state=None,
            max_iter=20000):

        super().__init__(kernel=kernel,
        alpha=alpha,
        optimizer=optimizer,
        n_restarts_optimizer=n_restarts_optimizer,
        normalize_y=normalize_y,
        copy_X_train=copy_X_train,
        n_targets=n_targets,
        random_state=random_state)

        self.max_iter = max_iter

    def _constrained_optimization(self, obj_func, initial_theta, bounds):
        global new_optimizer
        def new_optimizer(obj_func, initial_theta, bounds):
            opt_res =  scipy.optimize.minimize(
                obj_func,
                initial_theta,
                method="L-BFGS-B",
                jac=True,
                bounds=bounds,
                options={'maxiter': self.max_iter}
            )
            _check_optimize_result("lbfgs", opt_res)
            theta_opt, func_min = opt_res.x, opt_res.fun
            return theta_opt, func_min
        self.optimizer = new_optimizer
        return super()._constrained_optimization(obj_func, initial_theta, bounds)

class MultiLinearGAM(BaseEstimator):
    def __init__(self, dim = 3):
        self.dim = dim
        self.gams = [LinearGAM() for _ in range(dim)]
    
    def fit(self, X, y, weights=None):
        assert self.dim == y.shape[-1]
        for i in range(self.dim):
            self.gams[i].fit(X, y[:, i], weights)
    
    def predict(self, X):
        y = []
        for g in self.gams:
            y.append(g.predict(X))
        return np.vstack(y).T

# metric:  {'correlation', 'sokalmichener', 'hamming', 'sqeuclidean', 'canberra',
#  'cosine', 'l1', 'sokalsneath', 'chebyshev', 'rogerstanimoto', 'nan_euclidean', 'minkowski',
# 'l2', 'braycurtis', 'russellrao', 'euclidean', 'manhattan', 'yule', 'dice', 'jaccard', 'cityblock'}
class IsoMapRegressor(BaseEstimator):
    def __init__(self, n_components=5):
        self.isomap = Isomap(n_components=n_components, metric='braycurtis')
        self.regressor = LinearRegression()
    
    def fit(self, X, y):
        X_embed = self.isomap.fit_transform(X)
        self.regressor.fit(X_embed, y)
    
    def predict(self, X):
        X_embed = self.isomap.transform(X)
        return self.regressor.predict(X_embed)

class LLERegressor(BaseEstimator):
    def __init__(self, n_components=5, n_neighbors=5):
        self.lle = LocallyLinearEmbedding(n_components=n_components, n_neighbors=n_neighbors, random_state=42)
        self.regressor = LinearRegression()
    
    def fit(self, X, y):
        X_embed = self.lle.fit_transform(X)
        self.regressor.fit(X_embed, y)
    
    def predict(self, X):
        X_embed = self.lle.transform(X)
        return self.regressor.predict(X_embed)

class SpectralRegressor(BaseEstimator):
    def __init__(self, n_components=5):
        self.spectral = SpectralEmbedding(n_components=n_components)
        self.regressor = LinearRegression()
    
    def fit(self, X, y):
        X_embed = self.spectral.fit_transform(X)
        self.regressor.fit(X_embed, y)
    
    def predict(self, X):
        X_embed = self.spectral.transform(X) # TODO: Implement the `transform` method
        return self.regressor.predict(X_embed)

class MultiSVR(MultiOutputRegressor):
    def __init__(self, **kwargs):
        super().__init__(SVR(**kwargs))

    def __getattr__(self, name):
        if name in self.estimator.get_params():
            return getattr(self.estimator, name)
        raise AttributeError(f"'{self.__class__.__name__}' object has no attribute '{name}'")

    def set_params(self, **params):
        estimator_params = {}
        for key, value in params.items():
            if key in self.estimator.get_params():
                estimator_params[key] = value
            else:
                super().set_params(**{key: value})
        
        if estimator_params:
            self.estimator.set_params(**estimator_params)
        return self

class MultiGBDT(MultiOutputRegressor):
    def __init__(self, **kwargs):
        super().__init__(GradientBoostingRegressor(**kwargs))

    def __getattr__(self, name):
        if name in self.estimator.get_params():
            return getattr(self.estimator, name)
        raise AttributeError(f"'{self.__class__.__name__}' object has no attribute '{name}'")

    def set_params(self, **params):
        estimator_params = {}
        for key, value in params.items():
            if key in self.estimator.get_params():
                estimator_params[key] = value
            else:
                super().set_params(**{key: value})
        
        if estimator_params:
            self.estimator.set_params(**estimator_params)
        return self

class MultiAdaBoost(MultiOutputRegressor):
    def __init__(self, **kwargs):
        super().__init__(AdaBoostRegressor(**kwargs))

    def __getattr__(self, name):
        if name in self.estimator.get_params():
            return getattr(self.estimator, name)
        raise AttributeError(f"'{self.__class__.__name__}' object has no attribute '{name}'")

    def set_params(self, **params):
        estimator_params = {}
        for key, value in params.items():
            if key in self.estimator.get_params():
                estimator_params[key] = value
            else:
                super().set_params(**{key: value})
        
        if estimator_params:
            self.estimator.set_params(**estimator_params)
        return self