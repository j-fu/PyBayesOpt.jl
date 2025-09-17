"""
    printstats(optimization)

Evaluate and print statistics of optimization.
"""
function printstats(bo)
    pbest, valbest = bestpoint(bo)
    pbest = round.(pbest, sigdigits = 5)
    valbest = round(valbest, sigdigits = 5)

    status = "???"
    if initializing(bo)
        status = "Initializing"
    elseif bo._initialization_complete && bo._optim_iterations_done == 0
        status = "Initialized"
    elseif optimizing(bo)
        status = "Optimizing"
    elseif finished(bo)
        status = "Finished"
    end

    println("$(status): best at $(pbest), value=$(valbest) ")

    if bo._optimization_complete
        meanbest, varbest = evalpost(bo, pbest)
        varbest = round(varbest, sigdigits = 5)
        meanbest = round(meanbest, sigdigits = 5)

        pmax, stddev = samplemaxpost(bo)
        stddev = round.(stddev, sigdigits = 5)
        pmax = round.(pmax, sigdigits = 5)

        meanmax, varmax = evalpost(bo, pmax)
        varmax = round(varmax, sigdigits = 5)
        meanmax = round(meanmax, sigdigits = 5)

        println("Posterior: max at $(pmax)  stddev=$(stddev)")
        println("  mean(best)=$(meanbest) variance=$(varbest)")
        println("  mean(max)=$(meanmax) variance=$(varmax)")
    end
    return nothing
end

"""
    bestpoint(optimizer)

Return the best point and function value from the observation set.
"""
function bestpoint(bo)
    p, v = py"bestpoint"(bo._gpmodel, bo._X_obs, bo._Y_obs)
    return _toscale!(copy(p), bo), v
end

"""
    evalpost(optimizer, point)
        b
Evaluate posterior at given point.
Returns posterior mean and variance.
"""
function evalpost(bo, p)
    @assert bo._optimization_complete
    mean, var = py"evalpost"(bo._gpmodel, py"totorch"(_to01!(copy(p), bo)))
    return mean, var
end

"""
    samplemaxpost(optimizer; nsamples)

Sample the posterior maximum using 
[MaxPosteriorSampling](https://botorch.readthedocs.io/en/stable/generation.html#botorch.generation.sampling.MaxPosteriorSampling).

Returns the estimated maximum point and the estimated standard deviation of its coordinates.
Use [`evalpost`](@Ref) to obtain the function value in that point.
"""
function samplemaxpost(bo; nsamples = 1000)
    @assert bo._optimization_complete
    maxcoord, stddev = py"samplemaxpost"(bo._gpmodel, nsamples)
    return _toscale!(maxcoord, bo), _toscale!(stddev, bo) - bo.bounds[:, 1]
end
