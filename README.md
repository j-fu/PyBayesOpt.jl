# PyBayesOpt.jl

Julia wrapper to some Bayesian optimization tools available in Python:

- [BayesianOptimization](https://github.com/bayesian-optimization/BayesianOptimization) aka [bayes_opt](https://bayesian-optimization.github.io/BayesianOptimization/3.1.0/reference/bayes_opt.html)
  or [bayesian-optimization](https://pypi.org/project/bayesian-optimization/) 
- A q-batch  Bayesian optimizer implemented using [BoTorch](https://botorch.org/)


It uses a very partial implementation of the optimizer interface provided by [Optim.jl](https://github.com/JuliaNLSolvers/Optim.jl).

## Prerequisites

The package uses PyCall.jl to access the python code, and therefore requires a python installation which is working well with this package.
See [requirements.txt](requirements.txt) for the python packages to be installed.

If you use this under linux, you may have to set the environment variable `TORCH_USE_RTLD_GLOBAL=1` in order to avoid some loading problems
with the mkl libray.
