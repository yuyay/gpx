from typing import Any, Union
import numpy as np
import torch
import gpytorch as gpt
from sklearn.base import RegressorMixin, BaseEstimator
from sklearn.metrics import r2_score

from .gpx_regressor_module import GPXRegressorModule
from .tensor_utils import to_ndarray, to_tensor


class GPXRegressor(RegressorMixin, BaseEstimator):
    """GPX for regression. 
    """
    def __init__(
        self, 
        kernel: Any = gpt.kernels.RBFKernel, 
        kernel_kwargs: dict = {}, 
        kernel_init_params: dict = {},
        max_iter: int = 150, 
        tol: float = 10**-4, 
        lr: float = 0.1,
        dtype: torch.dtype = torch.double, 
        verbose: bool = False,
    ):
        self.kernel = kernel
        self.kernel_kwargs = kernel_kwargs
        self.kernel_init_params = kernel_init_params
        self.max_iter = max_iter
        self.tol = tol 
        self.lr = lr
        self.dtype = dtype
        self.verbose = verbose 

        self.model = GPXRegressorModule(kernel, kernel_kwargs=kernel_kwargs, dtype=dtype)
        for k, v in kernel_init_params.items():
            self.model.kernel_obj.base_kernel.__dict__[k] = v


    def fit(self, X: np.ndarray, y: np.ndarray, Z: Union[np.ndarray, None]):
        """Train model.

        Parameters
        ----------
        X : numpy.ndarray
        y : numpy.ndarray
        Z : numpy.ndarray or None
        """
        Z = X if Z is None else Z
        X, y, Z = map(lambda x: x.type(self.dtype), to_tensor(X, y, Z))

        self.model.type(self.dtype)
        self.model.train_initialize(X, y, Z)
        self.model.train()
        if self.verbose:
            for param_name, param in self.model.named_parameters():
                print(f'Parameter name: {param_name:42} value = {param.item()}')

        # use LBFGS as optimizer since we can load the whole data to train
        def closure():
            optimizer.zero_grad()
            loss = self.model(X, y, Z)
            if self.verbose:
                print('Iter {0:3d}: loss ='.format(optimizer.iter_count), loss.item())
            optimizer.iter_count += 1
            loss.backward()
            return loss

        optimizer = torch.optim.LBFGS(
            self.model.parameters(), lr=self.lr, tolerance_change=self.tol, max_iter=self.max_iter)
        optimizer.iter_count = 1
        optimizer.step(closure)

        if self.verbose:
            for param_name, param in self.model.named_parameters():
                print(f'Parameter name: {param_name:42} value = {param.item()}')

        self.model.prepare_eval()
        return self


    def predict(
        self, X: np.ndarray, Z: Union[np.ndarray, None] = None, return_weights: bool = False
    ):
        """Prediction.

        Parameters
        ----------
        X : numpy.ndarray
        Z : numpy.ndarray or None
        return_weights : bool
            Decide whether to return sample-wise weights. 
        """

        Z = X if Z is None else Z
        X, Z = map(lambda x: x.type(self.dtype), to_tensor(X, Z))
        y_mean, y_cov = to_ndarray(*self.model.predict_targets(X, Z))
        if return_weights:
            w_mean, w_cov = to_ndarray(*self.model.predict_weights(X, Z))
            return y_mean, y_cov, w_mean, w_cov
        else:
            return y_mean, y_cov


    def score(
        self, X: np.ndarray, y: np.ndarray, Z: Union[np.ndarray, None] = None, 
        sample_weight: Union[np.ndarray, None] = None
    ):
        y_pred, _ = self.predict(X, Z)
        return r2_score(y, y_pred, sample_weight=sample_weight)
