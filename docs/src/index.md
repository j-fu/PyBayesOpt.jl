# PyBayesOpt.jl Documentation

Julia wrapper for Bayesian optimization tools available in Python.

```@setup index
using Optim
```

## Overview

PyBayesOpt.jl provides Julia interfaces to popular Python Bayesian optimization libraries:

- [BayesianOptimization](https://github.com/bayesian-optimization/BayesianOptimization) - A Python library for Bayesian optimization
- [BoTorch](https://botorch.org/) - A Bayesian optimization library built on PyTorch for q-batch optimization

The package implements the [Optim.jl](https://github.com/JuliaNLSolvers/Optim.jl) interface for seamless integration with Julia's optimization ecosystem.


## Quick Start

### Basic BoTorch Q-Batch Optimization

```julia
using PyBayesOpt
using Optim

# Define a test function to minimize
objective(x) = (x[1] - 2)^2 + (x[2] - 1)^2

# Set up optimization parameters
params = BoTorchQBatch(
    bounds = [-5.0 5.0; -5.0 5.0],  # [min max] for each dimension  
    nbatch = 4,     # batch size
    ninit = 10,     # initialization iterations
    nopt = 20       # optimization iterations
)

# Run optimization using Optim.jl interface
result = optimize(objective, params)

# Access results
println("Best point: ", result.minimizer)
println("Best value: ", result.minimum)
println("Function evaluations: ", result.f_calls)
```

### BayesianOptimization Wrapper

```julia
# Using the BayesianOptimization wrapper
params = BayesianOptimization(
    bounds = [-5.0 5.0; -5.0 5.0],
    ninit = 10,
    nopt = 50
)

result = optimize(objective, params)
```

### Interactive Optimization Loop

For more control over the optimization process:

```julia
# Interactive optimization with BoTorch
state = BoTorchQBatchState(params=BoTorchQBatch(bounds=[-5.0 5.0; -5.0 5.0]))

while !finished(state)
    # Get next batch of points to evaluate
    pts = ask!(state)
    
    # Evaluate function (can be parallelized)
    values = [objective(pts[:, i]) for i in 1:size(pts, 2)]
    
    # Provide results back to optimizer
    tell!(state, pts, values)
    
    # Check current best
    if state.optimization_complete
        best_pt, best_val = bestpoint(state)
        println("Current best: $best_pt -> $best_val")
    end
end
```

## Benchmark Functions

The package includes several standard optimization benchmarks:

```julia
# Use benchmark functions
branin = BraninFunction()
result = optimize(branin, BoTorchQBatch(bounds=branin.bounds))

ackley = AckleyFunction(dim=3)  # 3D Ackley function
result = optimize(ackley, BoTorchQBatch(bounds=ackley.bounds))

rosenbrock = RosenbrockFunction(dim=2)
result = optimize(rosenbrock, BoTorchQBatch(bounds=rosenbrock.bounds))
```

## Using Optim.jl Interface

PyBayesOpt implements the Optim.jl `optimize` function interface, making it a drop-in replacement for other Optim.jl optimizers:

```julia
using Optim
using PyBayesOpt

# Your objective function
function objective(x)
    return sum((x .- [2.0, 3.0]).^2)  # minimize at [2.0, 3.0]
end

# Create method instance
method = BoTorchQBatch(
    bounds = [0.0 5.0; 0.0 5.0],
    nbatch = 2,
    ninit = 8,
    nopt = 15,
    acqmethod = :qEI
)

# Use with Optim.optimize
result = Optim.optimize(objective, method)

# Standard Optim.jl result interface
println("Minimizer: ", Optim.minimizer(result))
println("Minimum: ", Optim.minimum(result))
println("F calls: ", Optim.f_calls(result))

# Additional methods from BoTorch
if isa(result, BoTorchQBatchState)
    # Evaluate posterior at any point
    mean, var = evalposterior(result, [2.0, 3.0])
    println("Posterior at [2,3]: mean=$mean, variance=$var")
    
    # Sample from posterior maximum distribution
    max_point, std_dev = sampleposteriormin(result, nsamples=1000)
    println("Estimated maximum: $max_point ± $std_dev")
end
```

## Advanced Usage

### Acquisition Functions

BoTorch supports various acquisition functions:

```julia
method = BoTorchQBatch(
    acqmethod = :qLogEI,  # Options: :qEI, :qNEI, :qLogEI, :qLogNEI, :qUCB, :qPI
    qUCB_beta = 2.5       # Only used for :qUCB
)
```

### Parallel Evaluation

The q-batch approach naturally supports parallel function evaluation:

```julia
using Distributed
addprocs(4)  # Add 4 worker processes

@everywhere using PyBayesOpt

function parallel_eval(objective, state)
    while !finished(state)
        pts = ask!(state)
        
        # Parallel evaluation using pmap
        values = pmap(i -> objective(pts[:, i]), 1:size(pts, 2))
        
        tell!(state, pts, values)
    end
    return state
end

result = parallel_eval(objective, BoTorchQBatchState(params=method))
```

## API Reference

```@docs
PyBayesOpt
PyBayesOpt.BoTorchQBatch
PyBayesOpt.BoTorchQBatchState
PyBayesOpt.BayesianOptimization
PyBayesOpt.initializing
PyBayesOpt.optimizing
PyBayesOpt.finished
PyBayesOpt.ask!
PyBayesOpt.tell!
PyBayesOpt.bestpoint
PyBayesOpt.evalposterior
PyBayesOpt.sampleposteriormin
PyBayesOpt.SimpleFunction
PyBayesOpt.BraninFunction
PyBayesOpt.AckleyFunction
PyBayesOpt.RosenbrockFunction
```

