"""
Placeholder for a short summary about BoTorchOpt.
"""
module BoTorchOpt
using PyCall

function __init__()
    return @pyinclude(joinpath(@__DIR__, "..", "pysrc", "botorchwrap.py"))
end

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


"""
   initializing(optimization)

Tell if optimization is in initialization state.
"""
function initializing(bo::BoTorchOptimization)
    return !bo._initialization_complete && bo._init_iterations_done < bo.ninit
end

"""
   optimizing(optimization)

Tell if optimization is in optimization loop.
"""
function optimizing(bo::BoTorchOptimization)
    return bo._initialization_complete && bo._optim_iterations_done < bo.nopt
end

"""
   optimizing(optimization)

Tell if optimization is finished
"""
function finished(bo::BoTorchOptimization)
    return bo._initialization_complete && bo._optimization_complete
end


function _toscale!(pts::AbstractMatrix, bo::BoTorchOptimization)
    (; bounds) = bo
    for i in 1:size(pts, 2)
        for j in 1:size(pts, 1)
            pts[j, i] = bounds[j, 1] + pts[j, i] * (bounds[j, 2] - bounds[j, 1])
        end
    end
    return pts
end

function _toscale!(pt::AbstractVector, bo::BoTorchOptimization)
    (; bounds) = bo
    for j in 1:size(pt, 1)
        pt[j] = bounds[j, 1] + pt[j] * (bounds[j, 2] - bounds[j, 1])
    end
    return pt
end

function _to01!(pts::AbstractMatrix, bo::BoTorchOptimization)
    (; bounds) = bo
    for i in 1:size(pts, 2)
        for j in 1:size(pts, 1)
            pts[j, i] = (pts[j, i] - bounds[j, 1]) / (bounds[j, 2] - bounds[j, 1])
        end
    end
    return pts
end

function _to01!(pt::AbstractVector, bo::BoTorchOptimization)
    (; bounds) = bo
    for j in 1:size(pt, 1)
        pt[j] = (pt[j] - bounds[j, 1]) / (bounds[j, 2] - bounds[j, 1])
    end
    return pt
end


function _generate_initial_candidates!(bo)
    (; nbatch, bounds, ninit, seed) = bo
    pts = py"generate_initial_candidates"(size(bounds, 1), nbatch * ninit, seed)'
    bo._X_ini = _toscale!(pts, bo)
    return nothing
end

"""
    ask!(optimization)

Ask for a new batch of points to be avaluated. Returns `dim x batchsize` matrix.
At once may generate intial candidates or optimize the acquisition.
"""
function ask!(bo::BoTorchOptimization)

    q = bo.nbatch
    if bo._init_iterations_done == 0 && isnothing(bo._X_ini)
        @assert size(bo.bounds, 2) == 2
        # !!! more consistency checks here
        _generate_initial_candidates!(bo)
    end

    if !bo._initialization_complete
        startbatch = bo._init_iterations_done * q + 1
        bo._init_iterations_done += 1
        if bo._init_iterations_done == bo.ninit
            bo._initialization_complete = true
        end
        return bo._X_ini[:, startbatch:(startbatch + q - 1)]
    end

    if !bo._optimization_complete
        acqf = py"create_acqf"(
            bo._gpmodel,
            string(bo.acqmethod),
            bo.qUCB_beta
        )

        xbounds = copy(bo.bounds)
        xbounds[:, 1] .= 0
        xbounds[:, 2] .= 1

        candidates = py"optimize_acquisition_function"(
            acqf,
            py"totorch"(Matrix(xbounds')),
            q,
            num_restarts = bo.acq_nrestarts,
            raw_samples = bo.acq_nsamples
        )
        bo._optim_iterations_done += 1
        if bo._optim_iterations_done >= bo.nopt
            bo._optimization_complete = true
        end
        return _toscale!(Matrix(candidates'), bo)
    end
    return nothing
end

"""
    tell!(optimization, candidates, valus)

Provide newly evaluated candidate points to the optimizatio, update the initialization resp. training set.    
"""
function tell!(bo::BoTorchOptimization, candidates, values)
    pts = _to01!(copy(candidates), bo)
    pycandidates = py"totorch"(pts')
    pyvalues = py"totorch"(values)

    if isnothing(bo._X_obs)
        bo._X_obs = pycandidates
        bo._Y_obs = pyvalues
    else
        bo._X_obs = py"torch.cat"([bo._X_obs, pycandidates])
        bo._Y_obs = py"torch.cat"([bo._Y_obs, pyvalues])
    end

    bo._evaluations_used += length(values)

    if bo._initialization_complete
        bo._gpmodel = py"fit_gp_model"(bo._X_obs, bo._Y_obs)
    end
    if bo.verbose
        printstats(bo)
    end
    return nothing
end


"""
    printstats(optimization)

Evaluate and print statistics of optimization.
"""
function printstats(bo)
    pbest, valbest = bestpoint(bo)
    pbest = round.(pbest, sigdigits = 5)
    valbest = round(valbest, sigdigits = 5)

    status = "???"
    if initializing(bo)
        status = "Initializing"
    elseif bo._initialization_complete && bo._optim_iterations_done == 0
        status = "Initialized"
    elseif optimizing(bo)
        status = "Optimizing"
    elseif finished(bo)
        status = "Finished"
    end

    println("$(status): best at $(pbest), value=$(valbest) ")

    if bo._optimization_complete
        meanbest, varbest = evalpost(bo, pbest)
        varbest = round(varbest, sigdigits = 5)
        meanbest = round(meanbest, sigdigits = 5)

        pmax, stddev = samplemaxpost(bo)
        stddev = round.(stddev, sigdigits = 5)
        pmax = round.(pmax, sigdigits = 5)

        meanmax, varmax = evalpost(bo, pmax)
        varmax = round(varmax, sigdigits = 5)
        meanmax = round(meanmax, sigdigits = 5)

        println("Posterior: max at $(pmax)  stddev=$(stddev)")
        println("  mean(best)=$(meanbest) variance=$(varbest)")
        println("  mean(max)=$(meanmax) variance=$(varmax)")
    end
    return nothing
end

"""
    bestpoint(optimizer)

Return the best point and function value from the observation set.
"""
function bestpoint(bo)
    p, v = py"bestpoint"(bo._gpmodel, bo._X_obs, bo._Y_obs)
    return _toscale!(copy(p), bo), v
end

"""
    evalpost(optimizer, point)

Evaluate posterior at given point.
Returns posterior mean and variance.
"""
function evalpost(bo, p)
    @assert bo._optimization_complete
    mean, var = py"evalpost"(bo._gpmodel, py"totorch"(_to01!(copy(p), bo)))
    return mean, var
end

"""
    samplemaxpost(optimizer; nsamples)

Sample the posterior maximum using 
[MaxPosteriorSampling](https://botorch.readthedocs.io/en/stable/generation.html#botorch.generation.sampling.MaxPosteriorSampling).

Returns the estimated maximum point and the estimated standard deviation of its coordinates.
Use [`evalpost`](@Ref) to obtain the function value in that point.
"""
function samplemaxpost(bo; nsamples = 100)
    @assert bo._optimization_complete
    maxcoord, stddev = py"samplemaxpost"(bo._gpmodel, nsamples)
    return _toscale!(maxcoord, bo), _toscale!(stddev, bo) - bo.bounds[:, 1]
end

"""
    optimize!(optimization, func)

Maximize black box function. Use multithreading if `Threads.nthreads()>1`.
"""
function optimize!(bo, func)
    while !finished(bo)
        pts = ask!(bo)
        values = zeros(bo.nbatch)
        if Threads.nthreads() > 1
            Threads.@threads for i in 1:size(pts, 2)
                values[i] = func(pts[:, i])
            end
        else
            for i in 1:size(pts, 2)
                values[i] = func(pts[:, i])
            end
        end
        tell!(bo, pts, values)
    end
    return nothing
end

export BoTorchOptimization, initializing, optimizing, finished, ask!, tell!, optimize!

end # module
