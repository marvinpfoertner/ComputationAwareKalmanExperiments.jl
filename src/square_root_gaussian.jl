struct SquareRootGaussian{Tm<:AbstractVector,TZ<:AbstractMatrix}
    m::Tm
    Z::TZ
end

function Base.:*(A::AbstractMatrix, u::SquareRootGaussian)
    return SquareRootGaussian(A * u.m, A * u.Z)
end

function Statistics.mean(u::SquareRootGaussian)
    return u.m
end

function Statistics.cov(u::SquareRootGaussian)
    return u.Z * u.Z'
end

function Statistics.var(u::SquareRootGaussian)
    return sum(u.Z .^ 2, dims = 2)[:, 1]
end

function Statistics.std(u::SquareRootGaussian)
    return sqrt.(Statistics.var(u))
end
