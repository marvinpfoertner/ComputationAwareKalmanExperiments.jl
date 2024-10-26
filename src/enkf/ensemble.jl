struct EnsembleGaussian{Tm<:AbstractVector,TZ<:AbstractMatrix}
    m::Tm
    Z::TZ
end

function members(u::EnsembleGaussian)
    return sqrt(size(u.Z, 2) - 1) * u.Z .+ u.m
end

