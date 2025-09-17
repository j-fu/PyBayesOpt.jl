"""
Placeholder for a short summary about BoTorchOpt.
"""
module BoTorchOpt
using PyCall

function __init__()
    return @pyinclude(joinpath(@__DIR__, "..", "pysrc", "botorchwrap.py"))
end

include("optimizationstruct.jl")
include("internals.jl")
include("postprocess.jl")
include("optimizationapi.jl")
export BoTorchOptimization, initializing, optimizing, finished, ask!, tell!, optimize!
export bestpoint, evalpost, samplemaxpost
include("benchmarks.jl")


end # module
