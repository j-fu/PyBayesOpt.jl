# PyBayesOpt.jl - Optim.jl Interface Examples
# ============================================

using PyBayesOpt
using Optim
using Printf

println("="^60)
println("PyBayesOpt.jl - Optim.jl Interface Examples")
println("="^60)

# Define a test function to minimize
function himmelblau(x)
    """Himmelblau's function - has 4 global minima"""
    return (x[1]^2 + x[2] - 11)^2 + (x[1] + x[2]^2 - 7)^2
end

println("\n1. Basic BoTorch Q-Batch Optimization")
println("-"^40)

# Create optimization method instance
method1 = BoTorchQBatch(
    bounds = [-5.0 5.0; -5.0 5.0],  # [min max] for each dimension
    nbatch = 3,      # batch size - evaluate 3 points simultaneously
    ninit = 8,       # initialization iterations: 8 * 3 = 24 initial evaluations
    nopt = 12,       # optimization iterations: 12 * 3 = 36 optimization evaluations
    acqmethod = :qLogEI,  # acquisition function
    verbose = true
)

# Use Optim.jl interface
result1 = Optim.optimize(himmelblau, method1)

# Access results using standard Optim.jl interface
@printf("Method: %s\n", typeof(result1.params))
@printf("Best point: [%.4f, %.4f]\n", result1.minimizer[1], result1.minimizer[2])
@printf("Best value: %.6f\n", result1.minimum)
@printf("Function evaluations: %d\n", result1.f_calls)

# BoTorch-specific analysis (if completed optimization)
if result1.optimization_complete
    println("\nBoTorch-specific analysis:")
    # Evaluate posterior at the best point
    mean, var = evalposterior(result1, result1.minimizer)
    @printf("Posterior at best point: mean=%.4f, variance=%.4f\n", mean, var)

    # Sample from posterior maximum distribution
    max_point, std_dev = sampleposteriormin(result1, nsamples = 500)
    @printf(
        "Estimated true optimum: [%.4f ± %.4f, %.4f ± %.4f]\n",
        max_point[1], std_dev[1], max_point[2], std_dev[2]
    )
end

println("\n" * "="^60)
println("\n2. BayesianOptimization Wrapper")
println("-"^40)

# Using the classical BayesianOptimization wrapper
method2 = BayesianOptimization(
    bounds = [-5.0 5.0; -5.0 5.0],
    ninit = 15,      # 15 initial random evaluations
    nopt = 35,       # 35 Bayesian optimization iterations
    verbose = 0      # Python library verbosity
)

result2 = Optim.optimize(himmelblau, method2)

@printf("Method: %s\n", typeof(result2.params))
@printf("Best point: [%.4f, %.4f]\n", result2.minimizer[1], result2.minimizer[2])
@printf("Best value: %.6f\n", result2.minimum)
@printf("Function evaluations: %d\n", result2.f_calls)

println("\n" * "="^60)
println("\n3. Benchmark Functions")
println("-"^40)

# Using benchmark functions
println("\nOptimizing Branin function:")
branin = BraninFunction()
result3 = Optim.optimize(
    branin, BoTorchQBatch(
        bounds = branin.bounds,
        nbatch = 2,
        ninit = 10,
        nopt = 15,
        verbose = false
    )
)

@printf(
    "Branin function - Best: [%.4f, %.4f] -> %.6f (true optimum: %.6f)\n",
    result3.minimizer[1], result3.minimizer[2], result3.minimum, branin.optimal_value
)

println("\nOptimizing 2D Ackley function:")
ackley = AckleyFunction(dim = 2)
result4 = Optim.optimize(
    ackley, BoTorchQBatch(
        bounds = ackley.bounds,
        nbatch = 4,
        ninit = 12,
        nopt = 20,
        verbose = false
    )
)

@printf(
    "Ackley function - Best: [%.4f, %.4f] -> %.6f (true optimum: %.6f)\n",
    result4.minimizer[1], result4.minimizer[2], result4.minimum, ackley.optimal_value
)

println("\n" * "="^60)
println("\n4. Custom Optimization Loop")
println("-"^40)

# Manual optimization loop for maximum control
function custom_optimization_demo()
    state = BoTorchQBatchState(
        params = BoTorchQBatch(
            bounds = [-3.0 3.0; -3.0 3.0],
            nbatch = 2,
            ninit = 6,
            nopt = 10,
            verbose = false
        )
    )

    iteration = 0
    while !finished(state)
        iteration += 1

        # Get next points to evaluate
        pts = ask!(state)

        # Evaluate function (could be parallelized)
        values = [himmelblau(pts[:, i]) for i in 1:size(pts, 2)]

        # Provide results back to optimizer
        tell!(state, pts, values)

        # Status update
        if initializing(state)
            @printf("Init %d: evaluated %d points\n", iteration, size(pts, 2))
        elseif optimizing(state)
            best_pt, best_val = bestpoint(state)
            @printf(
                "Opt %d: best so far [%.4f, %.4f] -> %.6f\n",
                iteration - state.params.ninit, best_pt[1], best_pt[2], best_val
            )
        end
    end

    return state
end

final_state = custom_optimization_demo()
@printf(
    "\nFinal result: [%.4f, %.4f] -> %.6f\n",
    final_state.minimizer[1], final_state.minimizer[2], final_state.minimum
)

println("\n" * "="^60)
println("\n5. Acquisition Function Comparison")
println("-"^40)

acquisition_functions = [:qEI, :qLogEI, :qUCB, :qPI]

for acq in acquisition_functions
    println("\nTesting acquisition function: $acq")

    method = BoTorchQBatch(
        bounds = [-2.0 2.0; -2.0 2.0],
        nbatch = 2,
        ninit = 8,
        nopt = 10,
        acqmethod = acq,
        qUCB_beta = 2.0,  # only used for qUCB
        verbose = false
    )

    result = Optim.optimize(x -> (x[1] - 1)^2 + (x[2] + 0.5)^2, method)
    @printf(
        "  %s: best=%.6f at [%.4f, %.4f] (%d evals)\n",
        acq, result.minimum, result.minimizer[1], result.minimizer[2], result.f_calls
    )
end

println("\n" * "="^60)
println("Examples completed!")
println("="^60)
