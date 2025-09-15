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
    X_obs::Union{Nothing, PyObject} = nothing
    X_ini::Union{Nothing, Matrix{Float64}} = nothing
    Y_obs::Union{Nothing, PyObject} = nothing
    model::Union{Nothing, PyObject} = nothing
    acquisition_type::String = "qLogEI"
    verbose::Bool = false
    evaluations_used::Int = 0
    initialization_complete::Bool = false
    optimization_complete::Bool = false
    init_iterations::Int = 0
    optim_iterations::Int = 0
    seed::Int = 1234
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

function generate_initial_candidates!(bo)
    (; batch_size, bounds, initial_iterations, seed) = bo
    npoints = batch_size * initial_iterations
    @assert size(bounds, 2) == 2
    dim = size(bounds, 1)
    pts = Matrix(py"generate_initial_candidates"(dim, npoints, seed)')
    @assert dim == size(pts, 1)

    # Scale pts to bounds
    for i in 1:size(pts, 1)
        for j in 1:dim
            pts[j, i] = bounds[j, 1] + pts[j, i] * (bounds[j, 2] - bounds[j, 1])
        end
    end
    bo.X_ini = pts
    return nothing
end

function ask!(bo::BoTorchOptimizer)
    (; bounds) = bo
    q = bo.batch_size
    if bo.init_iterations == 0 && isnothing(bo.X_ini)
        generate_initial_candidates!(bo)
    end

    if !bo.initialization_complete
        startbatch = bo.init_iterations * q + 1
        pts = bo.X_ini[:, startbatch:(startbatch + q - 1)]
        bo.init_iterations += 1
        if bo.init_iterations == bo.initial_iterations
            bo.initialization_complete = true
        end
        return pts
    end

    if !bo.optimization_complete
        num_restarts = 20
        raw_samples = 512
        acqf = py"create_acqf"(
            bo.model,
            bo.acquisition_type,
            bo.beta
        )
        xbounds = [0 0; 1.0 1.0]

        candidates = py"optimize_acquisition_function"(
            acqf,
            py"totorch"(xbounds),
            q,
            num_restarts,
            raw_samples,
        )

        bo.optim_iterations += 1
        if bo.optim_iterations >= bo.optimization_iterations
            bo.optimization_complete = true
        end

        pts = Matrix(candidates')

        for i in 1:size(pts, 2)
            for j in 1:size(pts, 1)
                pts[j, i] = bounds[j, 1] + pts[j, i] * (bounds[j, 2] - bounds[j, 1])
            end
        end
        return pts
    end
    return nothing
end


function tell!(bo::BoTorchOptimizer, candidates, values)
    (; bounds) = bo
    dim = size(bounds, 1)
    pts = copy(candidates)
    for i in 1:size(pts, 1)
        for j in 1:dim
            pts[j, i] = (pts[j, i] - bounds[j, 1]) / (bounds[j, 2] - bounds[j, 1])
        end
    end
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

    return nothing
end


function best(bo)
    (; bounds) = bo
    pt, val = py"get_best"(bo.X_obs, bo.Y_obs)
    dim = size(bounds, 1)
    for j in 1:dim
        pt[j] = bounds[j, 1] + pt[j] * (bounds[j, 2] - bounds[j, 1])
    end
    return pt, val
end

function optimize!(bo, func)
    while !finished(bo)
        pts = ask!(bo)
        q = bo.batch_size
        values = zeros(q)
        for i in 1:q
            values[i] = func(pts[:, i])
        end
        tell!(bo, pts, values)
        pt, v = best(bo)
    end
    return best(bo)
end
export BoTorchOptimizer, initializing, optimizing, finished, ask!, tell!, optimize!

end # module
