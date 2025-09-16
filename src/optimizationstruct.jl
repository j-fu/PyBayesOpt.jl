"""
    struct BoTorchOptimization

Struct describing optimization and its state. It just can be used once.
For repeated optimizations create new instances.


## Fields: 

### Method parameters and their defaults:
- `bounds::Matrix{Float64} = [-1 1]\'` : `ndim x 2` matrix of evaluation bounds
- `nbatch::Int = 1`: batch size for evaluations of black box model
- `ninit::Int = 10`: number of initialization iterations resulting in `nbatch*ninit` evaluations
- `nopt::Int = 10`: number of optimization iterations resulting in `nbatch*nopt` evaluations
- `acqmethod::Symbol = :qLogEI`: acquisition method.  

   Valid metods:
    - `:qEI`, `:qExpectedImprovement`
    - `:qLogEI`, `:qLogExpectedImprovement`
    - `:qUCB`, `:qUpperConfidenceBound`
    - `:qPI`, `:qProbabilityOfImprovement`
- `seed::Int = 1234`: random seed
- `verbose::Bool = true`: verbosity
- `acq_nrestarts::Int = 20`: `num_restarts` parameter in `optimize_acqf`
- `acq_nsamples::Int = 512`: `raw_samples` parameter in  `optimize_acqf`
- `qUCB_beta::Float64 = 2.0`: beta parameter for qUCB_beta acquisition method

### Internal state:
- `_X_ini::Union{Nothing, Matrix{Float64}} = nothing`: initialization points
- `_X_obs::Union{Nothing, PyObject} = nothing`: training points
- `_Y_obs::Union{Nothing, PyObject} = nothing`: training values
- `_gpmodel::Union{Nothing, PyObject} = nothing`: Gaussian process model
- `_evaluations_used::Int = 0`: number of evaluations done
- `_initialization_complete::Bool = false`: flag indicating initialization state
- `_optimization_complete::Bool = false`: flag indicating optimization state
- `_init_iterations_done::Int = 0`: initial iterations performed
- `_optim_iterations_done::Int = 0`: optimization iterations performed
"""
Base.@kwdef mutable struct BoTorchOptimization
    bounds::Matrix{Float64} = [-1 1]'
    nbatch::Int = 4
    ninit::Int = 4
    nopt::Int = 8
    seed::Int = 1234
    verbose::Bool = true
    acqmethod::Symbol = :qLogEI
    acq_nrestarts::Int = 20
    acq_nsamples::Int = 512
    qUCB_beta::Float64 = 2.0
    _X_ini::Union{Nothing, Matrix{Float64}} = nothing
    _X_obs::Union{Nothing, PyObject} = nothing
    _Y_obs::Union{Nothing, PyObject} = nothing
    _gpmodel::Union{Nothing, PyObject} = nothing
    _evaluations_used::Int = 0
    _initialization_complete::Bool = false
    _optimization_complete::Bool = false
    _init_iterations_done::Int = 0
    _optim_iterations_done::Int = 0
end
