using BoTorchOpt
using Test

function black_box_function(x)
    return -x[1]^2 - (x[2] - 1)^2 + 1
end


function test1(; q = 1)

    bo = BoTorchOptimizer(;
        bounds = [-10 -10; 10 10]',
        seed = rand(10:1000),
        batch_size = q
    )

    return optimize!(bo, black_box_function)
end
