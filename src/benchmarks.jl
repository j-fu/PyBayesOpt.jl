"""
    abstract type AbstractBenchmarkFunction

Abstract base type for benchmark optimization functions.

All benchmark functions should implement:
- Function call syntax `f(x)` where `x` is a vector
- Fields `bounds`, `optimal_value`, and optionally `optimal_point`
"""
abstract type AbstractBenchmarkFunction end

"""
    SimpleFunction <: AbstractBenchmarkFunction

A simple quadratic test function for optimization benchmarking.

Function: `f(x) = x[1]² + (x[2] - 1)² - 1`

# Fields
- `bounds::Matrix{Float64}`: Optimization bounds `[[-10 10]; [-10 10]]`
- `optimal_value::Float64`: Known optimal value `-1.0`
- `optimal_point::Vector{Float64}`: Known optimal point `[0.0, 1.0]`

# Example
```julia
simple_func = SimpleFunction()
result = optimize(simple_func, BoTorchQBatch(bounds=simple_func.bounds))
```
"""
Base.@kwdef struct SimpleFunction <: AbstractBenchmarkFunction
    bounds = [[-10 10.0]; [-10 10]]
    optimal_value = -1.0
    optimal_point = [0.0, 1.0]
end

function (::SimpleFunction)(x)
    return x[1]^2 + (x[2] - 1)^2 - 1
end

"""
    BraninFunction <: AbstractBenchmarkFunction

The Branin function - a common benchmark for global optimization.

Function: `f(x₁,x₂) = a(x₂ - bx₁² + cx₁ - r)² + s(1-t)cos(x₁) + s`
where `a=1, b=5.1/(4π²), c=5/π, r=6, s=10, t=1/(8π)`

# Fields  
- `bounds::Matrix{Float64}`: Optimization bounds `[[-5 10]; [0 15]]`
- `optimal_value::Float64`: Global minimum value `0.397887`

The function has 3 global minima at approximately:
- `(-π, 12.275)`, `(π, 2.275)`, `(9.42478, 2.475)`

# Example
```julia
branin_func = BraninFunction()
result = optimize(branin_func, BoTorchQBatch(bounds=branin_func.bounds))
```
"""
Base.@kwdef struct BraninFunction <: AbstractBenchmarkFunction
    bounds = [[-5.0 10.0]; [0.0 15.0]]
    optimal_value = 0.397887
end

function (::BraninFunction)(x)
    x1 = x[1]
    x2 = x[2]
    b = 5.1 / (4 * pi^2)
    c = 5 / pi
    r = 6
    a = 1
    s = 10
    t = 1 / (8 * pi)
    term1 = a * (x2 - b * x1^2 + c * x1 - r)^2
    term2 = s * (1 - t) * cos(x1)
    y = term1 + term2 + s
    return y
end


"""
    AckleyFunction <: AbstractBenchmarkFunction

The Ackley function - a widely used benchmark function for global optimization.

Function: `f(x) = -a·exp(-b·√(1/d·∑xᵢ²)) - exp(1/d·∑cos(c·xᵢ)) + a + e`
where typically `a=20, b=0.2, c=2π`

# Fields
- `dim::Int`: Problem dimension (default: 2)  
- `bounds::Matrix{Float64}`: Optimization bounds (default: `[-32.768, 32.768]` for each dimension)
- `optimal_value::Float64`: Global minimum value `0.0`
- `optimal_point::Vector{Float64}`: Global minimum at origin

The function has a global minimum at the origin with many local minima.

# Example
```julia
ackley_func = AckleyFunction(dim=5)  # 5-dimensional Ackley function
result = optimize(ackley_func, BoTorchQBatch(bounds=ackley_func.bounds))
```
"""
Base.@kwdef struct AckleyFunction <: AbstractBenchmarkFunction
    dim = 2
    bounds = [ fill(-32.768, dim) fill(32.768, dim)]
    optimal_value = 0.0
    optimal_point = zeros(dim)
end


function (::AckleyFunction)(x)
    a, b, c = 20.0, 0.2, 2.0 * π
    len_recip = inv(length(x))
    sum_sqrs = zero(eltype(x))
    sum_cos = sum_sqrs
    for i in x
        sum_cos += cos(c * i)
        sum_sqrs += i^2
    end
    return -(
        -a * exp(-b * sqrt(len_recip * sum_sqrs)) -
            exp(len_recip * sum_cos) + a + ℯ
    )
end


"""
    RosenbrockFunction <: AbstractBenchmarkFunction

The Rosenbrock function (also known as "Rosenbrock's valley" or "banana function").

Function: `f(x) = ∑[100(xᵢ₊₁ - xᵢ²)² + (1 - xᵢ)²]` for i=1 to n-1

# Fields
- `dim::Int`: Problem dimension (default: 2)
- `bounds::Matrix{Float64}`: Optimization bounds (default: `[-5, 10]` for each dimension)  
- `optimal_value::Float64`: Global minimum value `0.0`
- `optimal_point::Vector{Float64}`: Global minimum at `ones(dim)`

The function has a global minimum in a narrow, curved valley making it challenging for optimization algorithms.

# Example
```julia
rosenbrock_func = RosenbrockFunction(dim=3)  # 3-dimensional Rosenbrock function
result = optimize(rosenbrock_func, BoTorchQBatch(bounds=rosenbrock_func.bounds))
```
"""
Base.@kwdef struct RosenbrockFunction
    dim = 2
    bounds = [ fill(-5.0, dim) fill(10.0, dim)]
    optimal_value = 0.0
    optimal_point = ones(dim)
end

function (::RosenbrockFunction)(x)
    result = zero(eltype(x))
    dim = length(x)
    for i in 1:(dim - 1)
        result += 100 * (x[i + 1] - x[i]^2)^2 + (1 - x[i])^2
    end
    return -result
end
