# Quick Start Example for PyBayesOpt.jl
# ====================================

using PyBayesOpt
using Optim: Optim, minimizer, f_calls

# Define a simple 2D function to minimize
function simple_quadratic(x)
    return (x[1] - 2.0)^2 + (x[2] - 1.0)^2 + 0.1
end

println("Optimizing simple quadratic function f(x) = (x₁-2)² + (x₂-1)² + 0.1")
println("True minimum: f(2, 1) = 0.1")
println()

# Method 1: BoTorch Q-Batch (recommended)
println("Method 1: BoTorch Q-Batch Optimization")
println("-"^40)

method1 = BoTorchQBatch(
    bounds = [0.0 4.0; -1.0 3.0],  # Search bounds
    nbatch = 3,                    # Evaluate 3 points per iteration
    ninit = 5,                     # 5 initialization iterations
    nopt = 8                       # 8 optimization iterations
)

result1 = optimize(simple_quadratic, method1)

println("Result:")
println("  Best point: $(minimizer(result1))")
println("  Best value: $(minimum(result1))")
println("  Total evaluations: $(f_calls(result1))")
println()

# Method 2: Classical BayesianOptimization
println("Method 2: Classical BayesianOptimization")
println("-"^42)

method2 = BayesianOptimization(
    bounds = [0.0 4.0; -1.0 3.0],
    ninit = 10,        # Initial random sampling
    nopt = 20          # Optimization iterations
)

result2 = optimize(simple_quadratic, method2)

println("Result:")
println("  Best point: $(minimizer(result2))")
println("  Best value: $(minimum(result2))")
println("  Total evaluations: $(f_calls(result2))")
println()
