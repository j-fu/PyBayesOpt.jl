"""
    struct BoTorchQBatch

Struct describing optimization for BoTorch based q-batch Bayesian
optimization

## Fields: 

- `bounds::Matrix{Float64} = [-1 1]\'` : `ndim x 2` matrix of evaluation bounds
- `nbatch::Int = 1`: batch size for evaluations of black box model
- `ninit::Int = 10`: number of initialization iterations resulting in `nbatch*ninit` evaluations
- `nopt::Int = 10`: number of optimization iterations resulting in `nbatch*nopt` evaluations
- `acqmethod::Symbol = :qLogEI`: acquisition method.  

   Valid metods (see [BoTorch docmentation](https://botorch.readthedocs.io/en/stable/acquisition.html)):
    - `:qEI`, `:qExpectedImprovement`
    - `:qNEI`, `:qNoisyExpectedImprovement`
    - `:qLogEI`, `:qLogExpectedImprovement`
    - `:qLogNEI`, `:qLogNoisyExpetedImprovement`
    - `:qUCB`, `:qUpperConfidenceBound`
    - `:qPI`, `:qProbabilityOfImprovement`

- `seed::Int = 1234`: random seed
- `verbose::Bool = true`: verbosity
- `acq_nrestarts::Int = 20`: `num_restarts` parameter in `optimize_acqf`
- `acq_nsamples::Int = 512`: `raw_samples` parameter in  `optimize_acqf`
- `qUCB_beta::Float64 = 2.0`: beta parameter for qUCB_beta acquisition method
"""
Base.@kwdef mutable struct BoTorchQBatch
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
end


"""
    struct BoTorchQBatchState

State for  [`BoTorchQBatch`](@ref), also used as result struct.

An instance of this state can be used either as the method paramer for the `optimize` function,
or in  user implemented loop as seen below:


    state = BoTorchQBatchState(; params)
    while !finished(state)
        pts = ask!(state)
        values = zeros(state.params.nbatch)
        Threads.@threads for i in 1:size(pts, 2)
              values[i] = func(pts[:, i])
        end
        tell!(state, pts, values)
    end


## Fields
- `X_ini::Union{Nothing, Matrix{Float64}} = nothing`: initialization points
- `X_obs::Union{Nothing, PyObject} = nothing`: training points
- `Y_obs::Union{Nothing, PyObject} = nothing`: training values
- `gpmodel::Union{Nothing, PyObject} = nothing`: Gaussian process model
- `evaluations_used::Int = 0`: number of evaluations done
- `initialization_complete::Bool = false`: flag indicating initialization state
- `optimization_complete::Bool = false`: flag indicating optimization state
- `init_iterations_done::Int = 0`: initial iterations performed
- `optim_iterations_done::Int = 0`: optimization iterations performed
"""
Base.@kwdef mutable struct BoTorchQBatchState <: Optim.OptimizationResults
    params::BoTorchQBatch = BoTorchQBatch()
    X_ini::Union{Nothing, Matrix{Float64}} = nothing
    X_obs::Union{Nothing, PyObject} = nothing
    Y_obs::Union{Nothing, PyObject} = nothing
    gpmodel::Union{Nothing, PyObject} = nothing
    initialization_complete::Bool = false
    optimization_complete::Bool = false
    init_iterations_done::Int = 0
    optim_iterations_done::Int = 0

    f_calls::Int = 0
    minimizer::Vector{Float64} = zeros(0)
    minimum::Float64 = 0
end

"""
    _toscale!(points::Matrix, state::BoTorchQBatchState)
    _toscale!(point::Vector, state::BoTorchQBatchState)
    
Scale point(s) from `[0,1]^d` to problem bounds.
"""
function _toscale! end

function _toscale!(pts::AbstractMatrix, state::BoTorchQBatchState)
    (; bounds) = state.params
    for i in 1:size(pts, 2)
        for j in 1:size(pts, 1)
            pts[j, i] = bounds[j, 1] + pts[j, i] * (bounds[j, 2] - bounds[j, 1])
        end
    end
    return pts
end

function _toscale!(pt::AbstractVector, state::BoTorchQBatchState)
    (; bounds) = state.params
    for j in 1:size(pt, 1)
        pt[j] = bounds[j, 1] + pt[j] * (bounds[j, 2] - bounds[j, 1])
    end
    return pt
end

"""
    _to01!(points::Matrix, state::BoTorchQBatchState)
    _to01!(point::Vector, state::BoTorchQBatchState)
    
Scale point(s) from problem bounds to  `[0,1]^d`.
"""
function _to01! end

function _to01!(pts::AbstractMatrix, state::BoTorchQBatchState)
    (; bounds) = state.params
    for i in 1:size(pts, 2)
        for j in 1:size(pts, 1)
            pts[j, i] = (pts[j, i] - bounds[j, 1]) / (bounds[j, 2] - bounds[j, 1])
        end
    end
    return pts
end

function _to01!(pt::AbstractVector, state::BoTorchQBatchState)
    (; bounds) = state.params
    for j in 1:size(pt, 1)
        pt[j] = (pt[j] - bounds[j, 1]) / (bounds[j, 2] - bounds[j, 1])
    end
    return pt
end


"""
   initializing(qbatch_state)

Tell if optimization is in initialization state.
"""
function initializing(state::BoTorchQBatchState)
    return !state.initialization_complete && state.init_iterations_done < state.params.ninit
end

"""
   optimizing(qbatch_state)

Tell if optimization is in optimization loop.
"""
function optimizing(state::BoTorchQBatchState)
    return state.initialization_complete && state.optim_iterations_done < state.params.nopt
end

"""
   finished(qbatch_state)

Tell if optimization is finished
"""
function finished(state::BoTorchQBatchState)
    return state.initialization_complete && state.optimization_complete
end

"""
    ask!(qbatch_state)

Ask for a new batch of points to be avaluated. Returns `dim x batchsize` matrix.
At once may generate intial candidates or optimize the acquisition.
"""
function ask!(state::BoTorchQBatchState)
    (; nbatch, bounds, ninit, seed) = state.params

    q = state.params.nbatch
    if state.init_iterations_done == 0 && isnothing(state.X_ini)
        @assert size(state.params.bounds, 2) == 2
        # !!! more consistency checks here
        pts = py"generate_initial_candidates"(size(bounds, 1), nbatch * ninit, seed)'
        state.X_ini = _toscale!(pts, state)
    end

    if !state.initialization_complete
        startbatch = state.init_iterations_done * q + 1
        state.init_iterations_done += 1
        if state.init_iterations_done == state.params.ninit
            state.initialization_complete = true
        end
        return state.X_ini[:, startbatch:(startbatch + q - 1)]
    end

    if !state.optimization_complete
        acqf = py"create_acqf"(
            state.gpmodel,
            string(state.params.acqmethod),
            state.params.qUCB_beta
        )

        xbounds = copy(state.params.bounds)
        xbounds[:, 1] .= 0
        xbounds[:, 2] .= 1

        candidates = py"optimize_acquisition_function"(
            acqf,
            py"totorch"(Matrix(xbounds')),
            q,
            num_restarts = state.params.acq_nrestarts,
            raw_samples = state.params.acq_nsamples
        )
        state.optim_iterations_done += 1
        if state.optim_iterations_done >= state.params.nopt
            state.optimization_complete = true
        end

        pts = _toscale!(Matrix(candidates'), state)
        return pts
    end
    return nothing
end

"""
    tell!(qbatch_state, candidates, valus)

Provide newly evaluated candidate points to the optimizatio, update the initialization resp. training set.    
"""
function tell!(state::BoTorchQBatchState, candidates, values)
    pts = _to01!(copy(candidates), state)
    pycandidates = py"totorch"(pts')
    pyvalues = py"totorch"(-values)

    if isnothing(state.X_obs)
        state.X_obs = pycandidates
        state.Y_obs = pyvalues
    else
        state.X_obs = py"torch.cat"([state.X_obs, pycandidates])
        state.Y_obs = py"torch.cat"([state.Y_obs, pyvalues])
    end

    state.f_calls += length(values)

    if state.initialization_complete
        state.gpmodel = py"fit_gp_model"(state.X_obs, state.Y_obs)
    end
    if state.optimization_complete
        state.minimizer, state.minimum = bestpoint(state)
    end
    if state.params.verbose
        summary(state)
    end
    return nothing
end

"""
    Optim.optimize(func, params::BoTorchQBatch)

Minimize black-box function `func` using the [`BoTorchQBatch`](@ref) Bayesian optimization method.

This provides an `Optim.jl`-compatible interface so you can call `optimize(func, method)` or
`Optim.optimize(func, method)` directly. The optimization proceeds in two phases:

1. Initialization: `params.ninit` batches of random (Sobol) points are evaluated.
2. Optimization: `params.nopt` iterations of model-based batch acquisition optimization.

Multi-threading: If `Threads.nthreads() > 1`, function evaluations within a batch are
threaded.

Arguments
- `func::Function`: Objective to minimize (should accept a vector of reals and return a scalar).
- `params::BoTorchQBatch`: Configuration object.

Returns
- `BoTorchQBatchState`: Final state which behaves like an `Optim.MultivariateOptimizationResults` object
  (supports `minimum`, `minimizer`, `f_calls`, etc.).
"""
function Optim.optimize(func, params::BoTorchQBatch)
    state = BoTorchQBatchState(; params)
    while !finished(state)
        pts = ask!(state)
        values = zeros(state.params.nbatch)
        if Threads.nthreads() > 1
            Threads.@threads for i in 1:size(pts, 2)
                values[i] = func(pts[:, i])
            end
        else
            for i in 1:size(pts, 2)
                values[i] = func(pts[:, i])
            end
        end
        tell!(state, pts, values)
    end
    return state
end


"""
    Optim.summary(state::BoTorchQBatchState)

Print a concise human-readable summary of a [`BoTorchQBatchState`](@ref) including phase,
current best point, best value, and (once optimization is complete) posterior statistics
at the incumbent.

Intended mainly for verbose / debugging output; called automatically when `params.verbose=true`.
"""
function Optim.summary(state::BoTorchQBatchState)
    pbest, valbest = bestpoint(state)
    pbest = round.(pbest, sigdigits = 5)
    valbest = round(valbest, sigdigits = 5)

    status = "???"
    if initializing(state)
        status = "Initializing"
    elseif state.initialization_complete && state.optim_iterations_done == 0
        status = "Initialized"
    elseif optimizing(state)
        status = "Optimizing"
    elseif finished(state)
        status = "Finished"
    end

    println("\n$(status): best at $(pbest), value=$(valbest) ")

    if state.optimization_complete
        meanbest, varbest = evalposterior(state, pbest)
        varbest = round(varbest, sigdigits = 5)
        meanbest = round(meanbest, sigdigits = 5)

        pmin, stddev = sampleposteriormin(state)
        stddev = round.(stddev, sigdigits = 5)
        pmin = round.(pmin, sigdigits = 5)

        meanmin, varmin = evalposterior(state, pmin)
        varmin = round(varmin, sigdigits = 5)
        meanmin = round(meanmin, sigdigits = 5)

        println("Posterior: min at $(pmin)  stddev=$(stddev)")
        println("  mean(best)=$(meanbest) variance=$(varbest)")
        println("  mean(min)=$(meanmin) variance=$(varmin)")
    end
    return nothing
end

function Base.show(io::IOContext, state::BoTorchQBatchState)
    show(io, state.params)
    return summary(state)
end

"""
    bestpoint(qbatch_state)

Return the best point and function value from the observation set.
"""
function bestpoint(state::BoTorchQBatchState)
    p, v = py"bestpoint"(state.gpmodel, state.X_obs, state.Y_obs)
    return _toscale!(copy(p), state), -v
end

"""
    evalposterior(qbatch_state, point)
        
Evaluate posterior at given point.
Returns posterior mean and variance.
"""
function evalposterior(state::BoTorchQBatchState, p)
    @assert state.optimization_complete
    mean, var = py"evalpost"(state.gpmodel, py"totorch"(_to01!(copy(p), state)))
    return -mean, var
end

"""
    sampleposteriormin(qbatch_state; nsamples)

Sample the posterior minimum using 
[MaxPosteriorSampling](https://botorch.readthedocs.io/en/stable/generation.html#botorch.generation.sampling.MaxPosteriorSampling).

Returns the estimated minimum point and the estimated standard deviation of its coordinates.
Use [`evalposterior`](@ref) to obtain the function value in that point.
"""
function sampleposteriormin(state::BoTorchQBatchState; nsamples = 1000)
    @assert state.optimization_complete
    maxcoord, stddev = py"samplemaxpost"(state.gpmodel, nsamples)
    return _toscale!(maxcoord, state), _toscale!(stddev, state) - state.params.bounds[:, 1]
end
