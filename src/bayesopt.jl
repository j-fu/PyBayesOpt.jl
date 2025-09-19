"""
    struct BayesianOptimization

Optimizer wrapping  [BayesianOptimization](https://github.com/bayesian-optimization/BayesianOptimization).

## Fields:
- 'bounds::Matrix{Float64}'
- 'ninit::Int = 10'
- 'nopt::Int = 100'
- 'verbose::Int = 0'
- 'seed::Int = 1'

"""
Base.@kwdef struct BayesianOptimization
    bounds::Matrix{Float64} = zeros(2, 2)
    ninit::Int = 10
    nopt::Int = 100
    verbose::Int = 0
    seed::Int = 1
end

"""
    struct BayesianOptimizationResult <: Optim.OptimizationResults

Result struct for [`BayesianOptimization`](@ref), compatible with Optim.jl interface.

# Fields
- `params::BayesianOptimization`: The optimization parameters used
- `f_calls::Int`: Number of function evaluations performed
- `minimizer::Vector{Float64}`: The best point found
- `minimum::Float64`: The best function value found
"""
Base.@kwdef mutable struct BayesianOptimizationResult <: Optim.OptimizationResults
    params::BayesianOptimization
    f_calls::Int = 0
    minimizer::Vector{Float64} = zeros(0)
    minimum::Float64 = 0
end

"""
    x00i(i)

Generate variable name in format "x001", "x002", etc. for variable index i.
"""
x00i(i) = @sprintf("x%03d", i)

"""
    get(d::Dict, i)

Get the i-th variable from dictionary d using the naming convention "x001", "x002", etc.
"""
get(d::Dict{Any, T}, i) where {T} = d[x00i(i)]
get(d::Dict{Symbol, T}, i) where {T} = d[Symbol(x00i(i))]

"""
    dict2vec(d)

Convert a dictionary with string keys "x001", "x002", ... to a vector.
"""
function dict2vec(d)
    v = [  get(d, i) for i in 1:length(d)]
    return v
end
"""
    vec2pairs(v)

Convert a vector to pairs suitable for the BayesianOptimization constructor.
"""
vec2pairs(v) = [ Pair(@sprintf("x%03d", i), v[i]) for i in 1:length(v)]

"""
    bounds2pairs(bounds::Matrix)

Convert a bounds matrix to pairs suitable for the BayesianOptimization constructor.
"""
function bounds2pairs(bounds::Matrix)
    return [ Pair(@sprintf("x%03d", i), (bounds[i, 1], bounds[i, 2])) for i in 1:size(bounds, 1)]
end


"""
    Optim.optimize(f, params::BayesianOptimization)

Minimize black-box function `f` using the Python `BayesianOptimization` library via the
[`BayesianOptimization`](@ref) parameter struct.

Workflow
1. Construct bounds dictionary.
2. Wrap objective as a maximization target (the Python library maximizes) by negating.
3. Run random initialization (`ninit`) followed by model-guided iterations (`nopt`).
4. Convert best (max) record back to a minimization result.

Returns a `BayesianOptimizationResult` with `minimum`, `minimizer`, and bookkeeping fields.
"""
function Optim.optimize(f, params::BayesianOptimization)
    xbounds = Dict(bounds2pairs(params.bounds))
    xf(; x...) = -f(dict2vec(Dict(x...)))
    optimizer = py"BayesianOptimization"(;
        f = xf,
        pbounds = xbounds,
        verbose = params.verbose,
        random_state = params.seed
    )
    optimizer.maximize(; init_points = params.ninit, n_iter = params.nopt)

    return BayesianOptimizationResult(;
        params,
        f_calls = length(optimizer.res),
        minimizer = dict2vec(Dict(pairs(optimizer.max["params"]))),
        minimum = -optimizer.max["target"]
    )

end
