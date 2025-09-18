"""
Placeholder for a short summary about BoTorchOpt.
"""
module BoTorchOpt
using PyCall
using Printf
using OrderedCollections

function __init__()
    @pyinclude(joinpath(@__DIR__, "..", "pysrc", "botorchwrap.py"))
    @pyinclude(joinpath(@__DIR__, "..", "pysrc", "bayesoptwrap.py"))
    return nothing
end

include("optimizationstruct.jl")
include("internals.jl")
include("postprocess.jl")
include("optimizationapi.jl")
export BoTorchOptimization, initializing, optimizing, finished, ask!, tell!, optimize!
export bestpoint, evalpost, samplemaxpost

include("bayesopt.jl")
export BayesianOptimization
include("benchmarks.jl")


end # module
