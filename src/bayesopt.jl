struct BayesianOptimization
    optimizer::PyObject
    ninit::Int
    nopt::Int
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


function BayesianOptimization(
        f = x -> x;
        bounds = zeros(2, 2),
        verbose = 0,
        random_state = 1,
        ninit::Int = 10,
        nopt::Int = 20
    )

    xbounds = Dict(bounds2pairs(bounds))
    xf(; x...) = f(dict2vec(Dict(x...)))
    return BayesianOptimization(
        py"BayesianOptimization"(; f = xf, pbounds = xbounds, verbose, random_state),
        ninit,
        nopt
    )
end

function maximize!(bo::BayesianOptimization)
    bo.optimizer.maximize(; init_points = bo.ninit, n_iter = bo.nopt)
    return bestpoint(bo)
end

function set_bounds!(bo::BayesianOptimization; new_bounds = (x = (0, 1)), lazy = true)
    return bo.optimizer.set_bounds(new_bounds = new_bounds, lazy = lazy)
end

function bestpoint(bo::BayesianOptimization)
    target = bo.optimizer.max["target"]
    params = bo.optimizer.max["params"]
    return dict2vec(Dict(pairs(params))), target
end

res(bo::BayesianOptimization) = bo.optimizer.res
iterations(bo::BayesianOptimization) = length(bo.optimizer.res)
