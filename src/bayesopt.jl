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
    struct BayesianOptimizationResult

Result struct for [`BayesianOptimization`](@ref)
"""
Base.@kwdef mutable struct BayesianOptimizationResult <: Optim.OptimizationResults
    params::BayesianOptimization
    f_calls::Int = 0
    minimizer::Vector{Float64} = zeros(0)
    minimum::Float64 = 0
end

x00i(i) = @sprintf("x%03d", i)
get(d::Dict{Any, T}, i) where {T} = d[x00i(i)]
get(d::Dict{Symbol, T}, i) where {T} = d[Symbol(x00i(i))]

function dict2vec(d)
    v = [  get(d, i) for i in 1:length(d)]
    return v
end
vec2pairs(v) = [ Pair(@sprintf("x%03d", i), v[i]) for i in 1:length(v)]
function bounds2pairs(bounds::Matrix)
    return [ Pair(@sprintf("x%03d", i), (bounds[i, 1], bounds[i, 2])) for i in 1:size(bounds, 1)]
end


"""
    optimize(func, params::BayesianOptimization)

Maximize black box function `func` using [`BayesianOptimization`](@ref) method
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
