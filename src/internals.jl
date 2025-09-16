function _toscale!(pts::AbstractMatrix, bo::BoTorchOptimization)
    (; bounds) = bo
    for i in 1:size(pts, 2)
        for j in 1:size(pts, 1)
            pts[j, i] = bounds[j, 1] + pts[j, i] * (bounds[j, 2] - bounds[j, 1])
        end
    end
    return pts
end

function _toscale!(pt::AbstractVector, bo::BoTorchOptimization)
    (; bounds) = bo
    for j in 1:size(pt, 1)
        pt[j] = bounds[j, 1] + pt[j] * (bounds[j, 2] - bounds[j, 1])
    end
    return pt
end

function _to01!(pts::AbstractMatrix, bo::BoTorchOptimization)
    (; bounds) = bo
    for i in 1:size(pts, 2)
        for j in 1:size(pts, 1)
            pts[j, i] = (pts[j, i] - bounds[j, 1]) / (bounds[j, 2] - bounds[j, 1])
        end
    end
    return pts
end

function _to01!(pt::AbstractVector, bo::BoTorchOptimization)
    (; bounds) = bo
    for j in 1:size(pt, 1)
        pt[j] = (pt[j] - bounds[j, 1]) / (bounds[j, 2] - bounds[j, 1])
    end
    return pt
end


function _generate_initial_candidates!(bo)
    (; nbatch, bounds, ninit, seed) = bo
    pts = py"generate_initial_candidates"(size(bounds, 1), nbatch * ninit, seed)'
    bo._X_ini = _toscale!(pts, bo)
    return nothing
end
