using BoTorchOpt
using BoTorchOpt: to01!, toscale!
using Test

function testscaling(dim, npoints)
    pts = rand(dim, npoints)
    bounds = zeros(dim, 2)
    bounds[:, 1] .= -10 * rand(dim)
    bounds[:, 2] .= 10 * rand(dim)

    return @test to01!(copy(toscale!(copy(pts), bounds)), bounds) ≈ pts
end


function black_box_function(x)
    return -x[1]^2 - (x[2] - 1)^2 + 1 - 1.0e2 * x[3]^2
end


function test1(; q = 1)

    bo = BoTorchOptimizer(;
        bounds = [-10 -10 -10; 10 10 10;]',
        seed = rand(10:1000),
        batch_size = q,
        acquisition_type = "qUCB"
    )

    return optimize!(bo, black_box_function)
end

test1(q = 3)
