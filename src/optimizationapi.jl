"""
   initializing(optimization)

Tell if optimization is in initialization state.
"""
function initializing(bo::BoTorchOptimization)
    return !bo._initialization_complete && bo._init_iterations_done < bo.ninit
end

"""
   optimizing(optimization)

Tell if optimization is in optimization loop.
"""
function optimizing(bo::BoTorchOptimization)
    return bo._initialization_complete && bo._optim_iterations_done < bo.nopt
end

"""
   optimizing(optimization)

Tell if optimization is finished
"""
function finished(bo::BoTorchOptimization)
    return bo._initialization_complete && bo._optimization_complete
end

"""
    ask!(optimization)

Ask for a new batch of points to be avaluated. Returns `dim x batchsize` matrix.
At once may generate intial candidates or optimize the acquisition.
"""
function ask!(bo::BoTorchOptimization)

    q = bo.nbatch
    if bo._init_iterations_done == 0 && isnothing(bo._X_ini)
        @assert size(bo.bounds, 2) == 2
        # !!! more consistency checks here
        _generate_initial_candidates!(bo)
    end

    if !bo._initialization_complete
        startbatch = bo._init_iterations_done * q + 1
        bo._init_iterations_done += 1
        if bo._init_iterations_done == bo.ninit
            bo._initialization_complete = true
        end
        return bo._X_ini[:, startbatch:(startbatch + q - 1)]
    end

    if !bo._optimization_complete
        acqf = py"create_acqf"(
            bo._gpmodel,
            string(bo.acqmethod),
            bo.qUCB_beta
        )

        xbounds = copy(bo.bounds)
        xbounds[:, 1] .= 0
        xbounds[:, 2] .= 1

        candidates = py"optimize_acquisition_function"(
            acqf,
            py"totorch"(Matrix(xbounds')),
            q,
            num_restarts = bo.acq_nrestarts,
            raw_samples = bo.acq_nsamples
        )
        bo._optim_iterations_done += 1
        if bo._optim_iterations_done >= bo.nopt
            bo._optimization_complete = true
        end
        return _toscale!(Matrix(candidates'), bo)
    end
    return nothing
end

"""
    tell!(optimization, candidates, valus)

Provide newly evaluated candidate points to the optimizatio, update the initialization resp. training set.    
"""
function tell!(bo::BoTorchOptimization, candidates, values)
    pts = _to01!(copy(candidates), bo)
    pycandidates = py"totorch"(pts')
    pyvalues = py"totorch"(values)

    if isnothing(bo._X_obs)
        bo._X_obs = pycandidates
        bo._Y_obs = pyvalues
    else
        bo._X_obs = py"torch.cat"([bo._X_obs, pycandidates])
        bo._Y_obs = py"torch.cat"([bo._Y_obs, pyvalues])
    end

    bo._evaluations_used += length(values)

    if bo._initialization_complete
        bo._gpmodel = py"fit_gp_model"(bo._X_obs, bo._Y_obs)
    end
    if bo.verbose
        printstats(bo)
    end
    return nothing
end

"""
    optimize!(optimization, func)

Maximize black box function. Use multithreading if `Threads.nthreads()>1`.
"""
function optimize!(bo, func)
    while !finished(bo)
        pts = ask!(bo)
        values = zeros(bo.nbatch)
        if Threads.nthreads() > 1
            Threads.@threads for i in 1:size(pts, 2)
                values[i] = func(pts[:, i])
            end
        else
            for i in 1:size(pts, 2)
                values[i] = func(pts[:, i])
            end
        end
        tell!(bo, pts, values)
    end
    return nothing
end
