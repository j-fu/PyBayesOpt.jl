"""
Placeholder for a short summary about BoTorchOpt.
"""
module BoTorchOpt
using PyCall

function __init__()
    return @pyinclude(joinpath(@__DIR__, "..", "pysrc", "botorchwrap.py"))
end

Base.@kwdef mutable struct BoTorchOptimizer
    bounds::Matrix{Float64} = [-1 1]'
    batch_size::Int = 1
    beta::Float64 = 2.0
    initial_iterations::Int = 10
    optimization_iterations::Int = 10
    acquisition_type::String = "qLogEI"
    num_restarts::Int = 20
    raw_samples::Int = 512
    seed::Int = 1234
    verbose::Bool = true
    X_obs::Union{Nothing, PyObject} = nothing
    X_ini::Union{Nothing, Matrix{Float64}} = nothing
    Y_obs::Union{Nothing, PyObject} = nothing
    model::Union{Nothing, PyObject} = nothing
    evaluations_used::Int = 0
    initialization_complete::Bool = false
    optimization_complete::Bool = false
    init_iterations::Int = 0
    optim_iterations::Int = 0
end

function initializing(bo::BoTorchOptimizer)
    return !bo.initialization_complete && bo.init_iterations < bo.initial_iterations
end

function optimizing(bo::BoTorchOptimizer)
    return bo.initialization_complete && bo.optim_iterations < bo.optimization_iterations
end

function finished(bo::BoTorchOptimizer)
    return bo.initialization_complete && bo.optimization_complete
end

function toscale!(pts::AbstractMatrix, bounds)
    for i in 1:size(pts, 2)
        for j in 1:size(pts, 1)
            pts[j, i] = bounds[j, 1] + pts[j, i] * (bounds[j, 2] - bounds[j, 1])
        end
    end
    return pts
end

function toscale!(pt::AbstractVector, bounds)
    for j in 1:size(pt, 1)
        pt[j] = bounds[j, 1] + pt[j] * (bounds[j, 2] - bounds[j, 1])
    end
    return pt
end

function to01!(pts::AbstractMatrix, bounds)
    for i in 1:size(pts, 2)
        for j in 1:size(pts, 1)
            pts[j, i] = (pts[j, i] - bounds[j, 1]) / (bounds[j, 2] - bounds[j, 1])
        end
    end
    return pts
end

function to01!(pt::AbstractVector, bounds)
    for j in 1:size(pt, 1)
        pt[j] = (pt[j] - bounds[j, 1]) / (bounds[j, 2] - bounds[j, 1])
    end
    return pt
end


function generate_initial_candidates!(bo)
    (; batch_size, bounds, initial_iterations, seed) = bo
    npoints = batch_size * initial_iterations
    @assert size(bounds, 2) == 2
    pts = py"generate_initial_candidates"(size(bounds, 1), npoints, seed)'
    bo.X_ini = toscale!(pts, bounds)
    return nothing
end

function ask!(bo::BoTorchOptimizer)

    q = bo.batch_size
    if bo.init_iterations == 0 && isnothing(bo.X_ini)
        generate_initial_candidates!(bo)
    end

    if !bo.initialization_complete
        startbatch = bo.init_iterations * q + 1
        bo.init_iterations += 1
        if bo.init_iterations == bo.initial_iterations
            bo.initialization_complete = true
        end
        return bo.X_ini[:, startbatch:(startbatch + q - 1)]
    end

    if !bo.optimization_complete
        acqf = py"create_acqf"(
            bo.model,
            bo.acquisition_type,
            bo.beta
        )

        xbounds = copy(bo.bounds)
        xbounds[:, 1] .= 0
        xbounds[:, 2] .= 1

        candidates = py"optimize_acquisition_function"(
            acqf,
            py"totorch"(Matrix(xbounds')),
            q,
            num_restarts = bo.num_restarts,
            raw_samples = bo.raw_samples
        )
        bo.optim_iterations += 1
        if bo.optim_iterations >= bo.optimization_iterations
            bo.optimization_complete = true
        end
        return toscale!(Matrix(candidates'), bo.bounds)
    end
    return nothing
end


function tell!(bo::BoTorchOptimizer, candidates, values)
    pts = to01!(copy(candidates), bo.bounds)
    pycandidates = py"totorch"(pts')
    pyvalues = py"totorch"(values)

    if isnothing(bo.X_obs)
        bo.X_obs = pycandidates
        bo.Y_obs = pyvalues
    else
        bo.X_obs = py"torch.cat"([bo.X_obs, pycandidates])
        bo.Y_obs = py"torch.cat"([bo.Y_obs, pyvalues])
    end

    if bo.initialization_complete
        bo.model = py"fit_gp_model"(bo.X_obs, bo.Y_obs)
    end
    if bo.verbose  &&  bo.initialization_complete
        if bo.initialization_complete && bo.optim_iterations == 0
            println("Initialization complete")
            printstats(bo)
        end
        if bo.optimization_complete
            println("Optimization complete")
            printstats(bo)
        end
    end
    return nothing
end

function printstats(bo)
    p, v = bestpoint(bo)
    p = round.(p, sigdigits = 5)
    v = round(v, sigdigits = 5)
    println("Best: $(v) at $(p)")

    if bo.optimization_complete
        mean, var = evalpost(bo, p)
        var = round(var, sigdigits = 5)
        mean = round(mean, sigdigits = 5)
        println("      postmean=$(mean) var=$(var)")

        meancoord, stddev = samplemaxpost(bo)
        mean, var = evalpost(bo, meancoord)
        var = round(var, sigdigits = 5)
        mean = round(mean, sigdigits = 5)
        stddev = round.(stddev, sigdigits = 5)
        meancoord = round.(meancoord, sigdigits = 5)


        println("Postmax: postmean=$(mean) at $(meancoord)")
        println("         stddev=$(stddev), var=$(var)")
    end
    return nothing
end

"""
    bestpoint(optimizer)

Return the best point and function value from the observation set.
"""
function bestpoint(bo)
    p, v = py"bestpoint"(bo.model, bo.X_obs, bo.Y_obs)
    return toscale!(copy(p), bo.bounds), v
end

"""
    evalpost(optimizer, point)

Evaluate posterior at given point.
Returns posterior mean and variance.
"""
function evalpost(bo, p)
    @assert bo.optimization_complete
    mean, var = py"evalpost"(bo.model, py"totorch"(to01!(copy(p), bo.bounds)))
    return mean, var
end

"""
    samplemaxpost(optimizer; nsamples)

Sample the posterior maximum using 
[MaxPosteriorSampling](https://botorch.readthedocs.io/en/stable/generation.html#botorch.generation.sampling.MaxPosteriorSampling).

Returns the estimated maximum point and the estimated standard deviation.
"""
function samplemaxpost(bo; nsamples = 100)
    @assert bo.optimization_complete
    @time maxcoord, stddev = py"samplemaxpost"(bo.model, nsamples)
    return toscale!(maxcoord, bo.bounds), toscale!(stddev, bo.bounds) - bo.bounds[:, 1]
end


function optimize!(bo, func)
    while !finished(bo)
        pts = ask!(bo)
        values = zeros(bo.batch_size)
        for i in 1:bo.batch_size
            values[i] = func(pts[:, i])
        end
        tell!(bo, pts, values)
    end
    meancoord, stddev = samplemaxpost(bo)
    meanvalue, var = evalpost(bo, meancoord)
    println("meancoord: $(meancoord), $(meanvalue), v=$(func(meancoord)), $(var)")
    return bestpoint(bo)
end

export BoTorchOptimizer, initializing, optimizing, finished, ask!, tell!, optimize!

end # module
