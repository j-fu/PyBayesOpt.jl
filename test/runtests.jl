using PyBayesOpt
using PyBayesOpt: _to01!, _toscale!
using Test


function testscaling(dim, npoints)
    pts = rand(dim, npoints)
    bounds = zeros(dim, 2)
    bounds[:, 1] .= -10 * rand(dim)
    bounds[:, 2] .= 10 * rand(dim)

    return @test _to01!(copy(_toscale!(copy(pts), bounds)), bounds) ≈ pts
end


function runbenchmarks(; q = 4, nopt = 10)
    funcs = [
        #PyBayesOpt.SimpleFunction(),
        #PyBayesOpt.BraninFunction(),
        PyBayesOpt.AckleyFunction(),
        #PyBayesOpt.RosenbrockFunction(dim = 4),
    ]

    for func in funcs
        bo = PyBayesOptimization(;
            bounds = func.bounds,
            seed = rand(10:1000),
            nbatch = q,
            nopt,
            acq_nsamples = 1024,
            acqmethod = :qLogNEI
        )
        optimize!(bo, func)
        pt, val = bestpoint(bo)
        defect = abs(val - func.optimal_value)
        println("$(typeof(func)): defect=$(defect)")
    end
    return nothing
end


runbenchmarks(q = 4)
