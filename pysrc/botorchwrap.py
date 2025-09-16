import sys
import numpy as np
import torch
from scipy.stats import qmc
from botorch.models import SingleTaskGP
from botorch.fit import fit_gpytorch_mll
from gpytorch.mlls import ExactMarginalLogLikelihood
from botorch.acquisition import qExpectedImprovement, qLogExpectedImprovement, qUpperConfidenceBound, qProbabilityOfImprovement
from botorch.acquisition import PosteriorMean
from botorch.optim import optimize_acqf
from botorch.posteriors import GPyTorchPosterior
from botorch.generation import MaxPosteriorSampling



def generate_initial_candidates(dim, n, seed):
    """
    Generate n sample points in [0,1]^dim
    """
    sampler = qmc.LatinHypercube(d=dim, seed=seed)
    return sampler.random(n)	

def totorch(array): 
    """
    Turn a numpy array into a torch tensor, as used by BoTorch
    """
    return torch.as_tensor(array, dtype=torch.float64, device=torch.device('cpu'))

def tojulia(tensor):
    """
    Turn a torch tensor into numpy array which can by transparently used by Julia
    """
    return tensor.detach().cpu().numpy()


def fit_gp_model(X: torch.Tensor, Y: torch.Tensor) -> SingleTaskGP:
    """
    Create Gaussian Process model
    """
    # Ensure Y is 2D
    if Y.dim() == 1:
        Y = Y.unsqueeze(-1)

    # https://botorch.readthedocs.io/en/latest/models.html#botorch.models.gp_regression.SingleTaskGP
    model = SingleTaskGP(X, Y)

    # https://docs.gpytorch.ai/en/v1.14/marginal_log_likelihoods.html#gpytorch.mlls.ExactMarginalLogLikelihood
    mll = ExactMarginalLogLikelihood(model.likelihood, model)

    # https://botorch.readthedocs.io/en/latest/fit.html#botorch.fit.fit_gpytorch_mll
    fit_gpytorch_mll(mll)

    return model


def create_acqf(model, acq_type, beta):
    if acq_type == "qEI" or acq_type == "qExpectedImprovement":
        best_f = model.train_targets.max()
        return qExpectedImprovement(model, best_f=best_f)

    elif acq_type == "qLogEI" or acq_type == "qLogExpectedImprovement":
        best_f = model.train_targets.max()
        return qLogExpectedImprovement(model, best_f=best_f)
  
    elif acq_type == "qUCB" or acq_type == "qUpperConfidenceBound":
        return qUpperConfidenceBound(model, beta=beta)
    
    elif acq_type == "qPI" or acq_type == "qProbabilityOfImprovement":
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
    # https://botorch.readthedocs.io/en/latest/optim.html#botorch.optim.optimize.optimize_acqf
    candidates, _ = optimize_acqf(
        acq_function=acquisition_func,
        bounds=bounds,
        q=q,
        num_restarts=num_restarts,
        raw_samples=raw_samples,
    )
    return candidates.detach().cpu().numpy()


def estimate_local_uncertainty(gp, point, epsilon=1e-5):
    """Estimate local uncertainty using finite differences"""
    point = point.clone().requires_grad_(True)
    
    # Get posterior at best point
    posterior = gp.posterior(point.unsqueeze(0))
    mean = posterior.mean
    
    # Compute gradient and Hessian
    gradient = torch.autograd.grad(mean.sum(), point)[0]
    
    # Compute Hessian using finite differences
    hessian = torch.zeros(point.shape[0], point.shape[0])
    for i in range(point.shape[0]):
        # Create perturbation
        perturbed_point = point.clone()
        perturbed_point[i] += epsilon
        
        # Compute gradient at perturbed point
        posterior_perturbed = gp.posterior(perturbed_point.unsqueeze(0))
        mean_perturbed = posterior_perturbed.mean
        gradient_perturbed = torch.autograd.grad(mean_perturbed.sum(), perturbed_point)[0]
        
        # Finite difference approximation of Hessian
        hessian[i] = (gradient-gradient_perturbed) / epsilon
    
    # Uncertainty estimate (inverse of Hessian)
    try:
        covariance = torch.inverse(hessian)
        std_dev = torch.sqrt(torch.diag(covariance))
    except:
        # If Hessian is singular, use a different approach
        std_dev = torch.ones_like(point) * float('nan')
    
    return std_dev.detach().cpu().numpy()


def bestpoint(model, X_obs,Y_obs):
    best_idx = Y_obs.argmax()
    best_point = X_obs[best_idx].detach().cpu().numpy()
    best_value = Y_obs[best_idx].item()
    return best_point, best_value

def evalpost(model, point):
    model.eval()
    posterior = model.posterior(point.unsqueeze(0))
    mean = posterior.mean
    variance = posterior.variance
    return mean.item(), variance.item()

def samplemaxpost(model, nsamples):
    # After completing optimization, with your trained GP model
    model.eval()
    # Method 1: Thompson sampling approach
    thompson_sampler = MaxPosteriorSampling(model=model, replacement=False)
    inputs=torch.as_tensor(model.train_inputs[0]) #.detach().cpu().numpy()
    # Generate samples of optimal locations
    optimal_samples = []
    for _ in range(nsamples):
        # Draw a sample from the GP posterior
        sample = thompson_sampler(inputs,num_samples=1)
        optimal_samples.append(sample)

    optimal_samples = torch.cat(optimal_samples, dim=0)

    # Calculate mean and standard deviation of optimal coordinates
    mean_optimal_coords = optimal_samples.mean(dim=0)
    std_optimal_coords = optimal_samples.std(dim=0)
    return mean_optimal_coords.detach().cpu().numpy(),std_optimal_coords.detach().cpu().numpy()
