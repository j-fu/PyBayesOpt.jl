"""
    ExampleJuggler

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
export BoTorchQBatch
export optimize
export initializing, optimizing, finished, ask!, tell!, optimize!
export bestpoint, evalpost, samplemaxpost


include("bayesopt.jl")
export BayesianOptimization
include("benchmarks.jl")


end # module
