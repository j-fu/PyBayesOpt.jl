abstract type AbstractBenchmarkFunction end

Base.@kwdef struct SimpleFunction <: AbstractBenchmarkFunction
    bounds = [[-10 10.0]; [-10 10]]
    optimal_value = 1.0
    optimal_point = [0.0, 1.0]
end

function (::SimpleFunction)(x)
    return -x[1]^2 - (x[2] - 1)^2 + 1
end

Base.@kwdef struct BraninFunction <: AbstractBenchmarkFunction
    bounds = [[-5.0 10.0]; [0.0 15.0]]
    optimal_value = -0.397887
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
    return -y
end


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
