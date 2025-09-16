using BoTorchOpt
using BoTorchOpt: _to01!, _toscale!
using Test

function testscaling(dim, npoints)
    pts = rand(dim, npoints)
    bounds = zeros(dim, 2)
    bounds[:, 1] .= -10 * rand(dim)
    bounds[:, 2] .= 10 * rand(dim)

    return @test _to01!(copy(_toscale!(copy(pts), bounds)), bounds) ≈ pts
end


function black_box_function(x)
    return -x[1]^2 - (x[2] - 1)^2 + 1
end


function test1(; q = 1)

    bo = BoTorchOptimization(;
        bounds = [-10 -10; 10 10;]',
        seed = rand(10:1000),
        nbatch = q,
        acqmethod = :qUCB
    )

    return optimize!(bo, black_box_function)
end

test1(q = 3)
