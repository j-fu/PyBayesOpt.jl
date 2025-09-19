# PyBayesOpt.jl

Julia wrapper to some Bayesian optimization tools available in Python:

- [BayesianOptimization](https://github.com/bayesian-optimization/BayesianOptimization) aka [bayes_opt](https://bayesian-optimization.github.io/BayesianOptimization/3.1.0/reference/bayes_opt.html)
  or [bayesian-optimization](https://pypi.org/project/bayesian-optimization/) 
- A q-batch  Bayesian optimizer implemented using [BoTorch](https://botorch.org/)

It uses a very partial implementation of the optimizer interface provided by [Optim.jl](https://github.com/JuliaNLSolvers/Optim.jl).

## Installation

### Python prerequisites
The package uses PyCall.jl to access the python code, and therefore requires a python installation which is working well with this package.
See [requirements.txt](requirements.txt) for the python packages to be installed.

```bash
pip install -r requirements.txt
```

If you use this under linux, you may have to set the environment variable `TORCH_USE_RTLD_GLOBAL=1` in order to avoid some loading problems
with the mkl library:

```bash
export TORCH_USE_RTLD_GLOBAL=1
```

### Installation via PackageNursery registry
The package can be installed with the Julia package manager in a standard way
For the time being, it is registered in the julia package registry [https://github.com/j-fu/PackageNursery](https://github.com/j-fu/PackageNursery)
To add the registry (needed only once), and to install the package, 
from the Julia REPL, type `]` to enter the Pkg REPL mode and run:

```
pkg> registry add https://github.com/j-fu/PackageNursery
```

Please be aware that adding a registry to your Julia installation requires to
trust the registry maintainer for handling things in a correct way. In particular,
the registry should not register higher versions of packages which are already
registered in the Julia General Registry. One can check this by visiting the above mentionend
github repository URL and inspecting the contents.

### Installation via  repository URL
```julia
using Pkg
Pkg.add(url="https://github.com/j-fu/PyBayesOpt.jl")
```

## Quick Start

### Basic Usage with Optim.jl Interface

```julia
using PyBayesOpt
using Optim

# Define a function to minimize
f(x) = (x[1] - 2.0)^2 + (x[2] - 1.0)^2

# Create optimizer - BoTorch q-batch is recommended
method = BoTorchQBatch(
    bounds = [0.0 4.0; -1.0 3.0],  # [min max] for each dimension
    nbatch = 4,      # evaluate 4 points per iteration
    ninit = 10,      # 10 initialization iterations  
    nopt = 15        # 15 optimization iterations
)

# Optimize using standard Optim.jl interface
result = optimize(f, method)

# Access results
println("Best point: ", result.minimizer)    # [2.0, 1.0]
println("Best value: ", result.minimum)      # ≈ 0.0
println("Evaluations: ", result.f_calls)     # 100 total evaluations
```

### Interactive Optimization Loop

For maximum control over the optimization process:

```julia
state = BoTorchQBatchState(params=method)

while !finished(state)
    # Get next batch of points to evaluate
    candidates = ask!(state)
    
    # Evaluate function (can be parallelized)  
    values = [f(candidates[:, i]) for i in 1:size(candidates, 2)]
    
    # Provide results back to optimizer
    tell!(state, candidates, values)
end

# Final result
best_point, best_value = bestpoint(state)
```

### Using Benchmark Functions

```julia
# Built-in benchmark functions
branin = BraninFunction()
result = optimize(branin, BoTorchQBatch(bounds=branin.bounds))

ackley = AckleyFunction(dim=5)  # 5-dimensional
result = optimize(ackley, BoTorchQBatch(bounds=ackley.bounds))
```

## Features

### Acquisition Functions

BoTorch supports multiple acquisition functions:
- `:qEI` / `:qExpectedImprovement` - Expected Improvement
- `:qLogEI` / `:qLogExpectedImprovement` - Log Expected Improvement (recommended)
- `:qNEI` / `:qNoisyExpectedImprovement` - Noisy Expected Improvement
- `:qLogNEI` / `:qLogNoisyExpectedImprovement` - Log Noisy Expected Improvement  
- `:qUCB` / `:qUpperConfidenceBound` - Upper Confidence Bound
- `:qPI` / `:qProbabilityOfImprovement` - Probability of Improvement

### Parallel Evaluation

The q-batch approach naturally supports parallel function evaluation:

```julia
using Base.Threads

# In the optimization loop:
values = zeros(nbatch)
@threads for i in 1:nbatch
    values[i] = expensive_function(candidates[:, i])
end
```

### Posterior Analysis

After optimization, you can analyze the Gaussian process posterior:

```julia
# Evaluate posterior at any point
mean, variance = evalposterior(result, [1.0, 2.0])

# Sample from posterior maximum distribution
max_point, std_dev = samplemaxpost(result, nsamples=1000)
println("Estimated global optimum: $max_point ± $std_dev")
```

## Optimizers

### BoTorchQBatch (Recommended)

Advanced q-batch Bayesian optimization using BoTorch:

```julia
BoTorchQBatch(
    bounds = [-1.0 1.0; -1.0 1.0],    # Optimization bounds
    nbatch = 4,                        # Batch size
    ninit = 10,                        # Initialization iterations
    nopt = 20,                         # Optimization iterations  
    acqmethod = :qLogEI,               # Acquisition function
    seed = 1234,                       # Random seed
    verbose = true,                    # Print progress
    acq_nrestarts = 20,                # Acquisition optimization restarts
    acq_nsamples = 512,                # Raw samples for acquisition optimization
    qUCB_beta = 2.0                    # Beta parameter for qUCB
)
```

### BayesianOptimization

Classical Bayesian optimization wrapper:

```julia
BayesianOptimization(
    bounds = [-1.0 1.0; -1.0 1.0],    # Optimization bounds
    ninit = 10,                        # Initial random samples
    nopt = 50,                         # Optimization iterations
    verbose = 0,                       # Verbosity level
    seed = 1                           # Random seed
)
```

## Examples

See the `examples/` directory for:
- `quick_start.jl` - Basic usage examples  
- `optim_interface_example.jl` - Comprehensive Optim.jl interface demo

## AI usage statement

Github copilot with Claude Sonnet 4 and GPT-5 was used to design an initial python version of the BoTorch
based algorithm, and to brush up documentation and testing infrastructure.

