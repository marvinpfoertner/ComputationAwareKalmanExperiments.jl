function ensemble_to_gaussian(members::AbstractMatrix)
    m = mean(members, dims = 2)[:, 1]
    Z = (members .- m) ./ sqrt(size(members, 2) - 1)

    return SquareRootGaussian(m, Z)
end

function ensemble(u::SquareRootGaussian)
    return sqrt(size(u.Z, 2) - 1) * u.Z .+ u.m
end
