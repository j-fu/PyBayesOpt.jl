"""
    PyBayesOpt
$(read(joinpath(@__DIR__, "..", "README.md"), String))
"""
module PyBayesOpt
using PyCall
using Printf
using Optim

function __init__()
    @pyinclude(joinpath(@__DIR__, "..", "pysrc", "botorchwrap.py"))
    @pyinclude(joinpath(@__DIR__, "..", "pysrc", "bayesoptwrap.py"))
    return nothing
end

include("botorchqbatch.jl")
export BoTorchQBatch, BoTorchQBatchState
export initializing, optimizing, finished, ask!, tell!, bestpoint, evalposterior, sampleposteriormin


include("bayesopt.jl")
export BayesianOptimization, BayesianOptimizationResult

include("benchmarks.jl")
export AbstractBenchmarkFunction, SimpleFunction, BraninFunction, AckleyFunction, RosenbrockFunction


end # module
