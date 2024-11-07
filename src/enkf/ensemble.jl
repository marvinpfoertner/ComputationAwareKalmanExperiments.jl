struct EnsembleGaussian{Tm<:AbstractVector,TZ<:AbstractMatrix}
    m::Tm
    Z::TZ
end

function EnsembleGaussian(members::AbstractMatrix)
    m = mean(members, dims = 2)[:, 1]
    Z = (members .- m) ./ sqrt(size(members, 2) - 1)

    return EnsembleGaussian(m, Z)
end

function members(u::EnsembleGaussian)
    return sqrt(size(u.Z, 2) - 1) * u.Z .+ u.m
end

