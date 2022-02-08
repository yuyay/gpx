from typing import Any, Union
import torch 
import torch.nn as nn
from torch.distributions import MultivariateNormal
import gpytorch as gpt

class GPXRegressorModule(nn.Module):
    """GPX Regression Module.

    Parameters
    ----------
    kernel : object
        A kernel function module implemented in gpytorch.kernels.
    kernel_kwargs : dict
        A dictionary of parameters for ``kernel``.
    dtype : object or string
        dtype.
    """
    def __init__(
        self, 
        kernel: Any = gpt.kernels.RBFKernel, 
        kernel_kwargs: dict = {}, 
        dtype: torch.dtype = torch.double
    ):
        super(GPXRegressorModule, self).__init__()
        self.dtype = dtype
        self.kernel = kernel
        self.kernel_kwargs = kernel_kwargs
        self.kernel_obj = gpt.kernels.ScaleKernel(kernel(**kernel_kwargs))
        self._init_noise_w = 0.1 
        self._init_noise_y = 0.1

    def train_initialize(
        self, X: torch.Tensor, Y: torch.Tensor, Z: torch.Tensor
    ):
        self.X_tr = X
        self.Y_tr = Y
        self.Z_tr = Z
        self.n_samples = X.size(0)
        self.x_dim = X.size(1)
        self.z_dim = Z.size(1)
        self._register_initialized_parameters(X, Y, Z)
        self._ZZ = torch.matmul(Z, torch.t(Z))  # (n, n)

    def _register_initialized_parameters(
        self, X: torch.Tensor, Y: torch.Tensor, Z: torch.Tensor
    ):
        noise_w = nn.Parameter(torch.tensor(self._init_noise_w, dtype=self.dtype))
        noise_y = nn.Parameter(torch.tensor(self._init_noise_y, dtype=self.dtype))
        self.register_parameter("noise_w", noise_w)
        self.register_parameter("noise_y", noise_y)

    def forward(
        self, X: torch.Tensor, Y: torch.Tensor, Z: torch.Tensor
    ):
        """Forward calculation (calculating negative log Gaussian likelihood).

        Parameters
        ----------
        X : torch.tensor
            training data of original representation with size of (n_samples, n_x_features).
        Y : torch.tensor
            training data of targets with size of (n_samples, 1).
        Z : torch.tensor
            training data of simplified representation with size of (n_samples, n_z_features).
        """
        ZZ = self._ZZ if hasattr(self, "_ZZ") else None
        loss = self.neg_log_likelihood(X, Y, Z, ZZ) / X.shape[0]
        return loss

    def prepare_eval(self):
        C = self._compute_covar_y(self.X_tr, self.Z_tr, self._ZZ)
        self.C_inv = torch.cholesky_inverse(torch.linalg.cholesky(C))
        K = self._compute_kernel_matrix(self.X_tr, self.Z_tr)  # (n, n)
        self.KIinv = torch.inverse(K + self.noise_w**2 * torch.eye(self.n_samples))  # (n, n)
        self.ZKZ = (K + self.noise_w**2 * torch.eye(self.n_samples, dtype=self.dtype)) * self._ZZ # (n, n)

    def _compute_kernel_matrix(
        self, X: torch.Tensor, Z: torch.Tensor, 
        X2: Union[torch.Tensor, None] = None, Z2: Union[torch.Tensor, None] = None
    ):
        S = self.kernel_obj(X, X2).evaluate().type(self.dtype)  # (n, n')
        return S

    def neg_log_likelihood(
        self, X: torch.Tensor, Y: torch.Tensor, Z: torch.Tensor, 
        ZZ: Union[torch.Tensor, None] = None
    ):
        """Calculating negative log Gaussian likelihood.
        """
        if ZZ == None:
            ZZ = torch.matmul(Z, torch.t(Z))  # (n, n)
        covar_y = self._compute_covar_y(X, Z, ZZ)
        b = torch.zeros(self.n_samples, dtype=self.dtype)
        mn = MultivariateNormal(b, covar_y)
        nll = -mn.log_prob(Y)
        return nll

    def _compute_covar_y(
        self, X: torch.Tensor, Z: torch.Tensor, ZZ: torch.Tensor
    ):
        n = Z.size(0)
        K = self._compute_kernel_matrix(X, Z)
        covar = self.noise_y**2 * torch.eye(n, dtype=self.dtype)
        covar += K * ZZ
        covar += self.noise_w**2 * torch.diag(ZZ.diagonal())
        return covar  # (n, n)

    def predict_targets(
        self, X_new: torch.Tensor, Z_new: torch.Tensor
    ):
        """Predicting target variables.
        """
        X_all = torch.cat((self.X_tr, X_new), 0)
        Z_all = torch.cat((self.Z_tr, Z_new), 0)
        ZZ_all = torch.matmul(Z_all, torch.t(Z_all))  # (n_all, n_all)
        C_all = self._compute_covar_y(X_all, Z_all, ZZ_all)
        k = C_all[self.n_samples:][:, :self.n_samples]  # (n_new, n)
        Ey = torch.matmul(torch.matmul(k, self.C_inv), self.Y_tr)
        C_remain = C_all[self.n_samples:][:, self.n_samples:]  # (n_new, n_new)
        Vy = C_remain - torch.matmul(torch.matmul(k, self.C_inv), torch.t(k))
        return Ey, Vy

    def predict_weights(
        self, X_new: torch.Tensor, Z_new: torch.Tensor
    ):
        """Estimating local weights.
        """
        # inverse = lambda x: torch.cholesky_inverse(torch.cholesky(x))
        # n_new = Z_new.size(0)
        X_all = torch.cat((self.X_tr, X_new), 0)
        Z_all = torch.cat((self.Z_tr, Z_new), 0)
        K_all = self._compute_kernel_matrix(X_all, Z_all)  # (n+n_new, n+n_new)
        K = K_all[:self.n_samples][:, :self.n_samples]  # (n, n)
        swI = self.noise_w**2 * torch.eye(self.n_samples, dtype=self.dtype)  # (n, n)
        K += swI
        k = K_all[self.n_samples:][:, :self.n_samples]  # (n_new, n)
        A = torch.matmul(k, self.KIinv)  # (n_new, n)
        KZ = torch.matmul(K.expand(self.z_dim, -1, -1), torch.diag_embed(torch.t(self.Z_tr)))  # (z_dim, n, n)
        AKZ = torch.bmm(A.expand(self.z_dim, -1, -1), KZ)  # (z_dim, n_new, n)
        # sy2I = self.noise_y**2 * torch.eye(self.n_samples, dtype=self.dtype)
        AKZDinvZKZ = torch.matmul(torch.matmul(AKZ, self.C_inv), self.ZKZ)  # (z_dim, n_new, n)

        M = self.noise_y**(-2) * torch.matmul((AKZ - AKZDinvZKZ), self.Y_tr)  # (z_dim, n_new)
        M = torch.t(M)  # (n_new, z_dim)

        c = torch.diagonal(K_all[self.n_samples:][:, self.n_samples:]) + self.noise_w**2 # (n_new,)
        c = torch.diag_embed(torch.t(c.expand(self.z_dim, -1)))  # (n_new, z_dim, z_dim)
        Ak = torch.matmul(A.unsqueeze(-2), k.unsqueeze(-1)).flatten()  # (n_new)
        Ak = torch.diag_embed(torch.t(Ak.expand(self.z_dim, -1)))  # (n_new, z_dim, z_dim)
        AK = torch.matmul(A, K)  # (n_new, n) 
        AKA = torch.matmul(AK.unsqueeze(-2), A.unsqueeze(-1)).flatten()  # (n_new)
        AKA = torch.diag_embed(torch.t(AKA.expand(self.z_dim, -1)))  # (n_new, z_dim, z_dim)
        AKZDinvZKA = torch.matmul(
            torch.matmul(AKZ.permute(1, 0, 2), self.C_inv), AKZ.permute(1, 2, 0))  # (n_new, z_dim, z_dim)
        V = c - Ak + AKA - AKZDinvZKA

        return M, V