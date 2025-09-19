using PyBayesOpt
using Optim
using Test


function runbenchmarks(; q = 4, nopt = 10)
    funcs = [
        PyBayesOpt.SimpleFunction(),
        PyBayesOpt.BraninFunction(),
        #PyBayesOpt.AckleyFunction(),
        #PyBayesOpt.RosenbrockFunction(dim = 4),
    ]

    for func in funcs
        optimizers = [
            BoTorchQBatch(;
                bounds = func.bounds,
                seed = rand(10:1000),
                nbatch = q,
                nopt,
                acq_nsamples = 1024,
                acqmethod = :qLogNEI
            ),
            BayesianOptimization(;
                bounds = func.bounds,
                seed = rand(10:1000)
            ),
        ]
        for optimizer in optimizers
            result = optimize(func, optimizer)
            val = Optim.minimum(result)
            defect = abs(val - func.optimal_value)
            println("$(typeof(optimizer))($(typeof(func))): defect=$(defect)")
            @test defect < 0.1
        end
    end
    return nothing
end

@testset "Benchmarks" begin
    runbenchmarks(q = 4)
end
