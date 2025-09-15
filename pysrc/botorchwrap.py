import sys
import numpy as np
import torch
from scipy.stats import qmc
from botorch.models import SingleTaskGP
from botorch.fit import fit_gpytorch_mll
from gpytorch.mlls import ExactMarginalLogLikelihood
from botorch.acquisition import qExpectedImprovement, qLogExpectedImprovement, qUpperConfidenceBound, qProbabilityOfImprovement
from botorch.optim import optimize_acqf


def generate_initial_candidates(dim, n, seed):
    """
    Generate n sample points in [0,1]^dim
    """
    sampler = qmc.LatinHypercube(d=dim, seed=seed)
    return sampler.random(n)	

def totorch(array):
    return torch.as_tensor(array, dtype=torch.float64, device=torch.device('cpu'))

def tojulia(array):
    return array.detach().cpu().numpy()


def fit_gp_model(X: torch.Tensor, Y: torch.Tensor) -> SingleTaskGP:
    # Ensure Y is 2D
    if Y.dim() == 1:
        Y = Y.unsqueeze(-1)
        # Create and fit model
        model = SingleTaskGP(X, Y)
    mll = ExactMarginalLogLikelihood(model.likelihood, model)
    fit_gpytorch_mll(mll)
    return model


def create_acqf(model, acq_type, beta):
    if acq_type == "qEI":
        best_f = model.train_targets.max()
        return qExpectedImprovement(model, best_f=best_f)

    elif acq_type == "qLogEI":
        best_f = model.train_targets.max()
        return qLogExpectedImprovement(model, best_f=best_f)
  
    elif acq_type == "qUCB":
        return qUpperConfidenceBound(model, beta=beta)
    
    elif acq_type == "qPI":
        best_f = model.train_targets.max()
        return qProbabilityOfImprovement(model, best_f=best_f)
    else:
        raise ValueError(f"Unknown acquisition function type: {acq_type}")

def optimize_acquisition_function(acquisition_func,
    bounds,
    q,
    num_restarts,
    raw_samples
):
    candidates, _ = optimize_acqf(
        acq_function=acquisition_func,
        bounds=bounds,
        q=q,
        num_restarts=num_restarts,
        raw_samples=raw_samples,
    )
    
    return candidates.detach().cpu().numpy()

def get_best(X_obs,Y_obs):
    best_idx = Y_obs.argmax()
    best_point = X_obs[best_idx].detach().cpu().numpy()
    best_value = Y_obs[best_idx].item()
    return best_point, best_value
