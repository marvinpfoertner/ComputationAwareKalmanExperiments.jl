struct Gaussian{Tm<:AbstractVector,TP<:Symmetric}
    m::Tm
    P::TP
end

function Gaussian(m::AbstractVector, P::AbstractMatrix)
    return Gaussian(m, Symmetric(hermitianpart(P)))
end

function Base.:*(A::AbstractMatrix, u::Gaussian)
    return Gaussian(A * u.m, hermitianpart!(A * u.P * A'))
end

Statistics.mean(u::Gaussian) = u.m
Statistics.cov(u::Gaussian) = u.P
Statistics.var(u::Gaussian) = diag(u.P)
Statistics.std(u::Gaussian) = sqrt.(Statistics.var(u))

struct SquareRootGaussian{Tm<:AbstractVector,TZ<:AbstractMatrix}
    m::Tm
    Z::TZ
end

function Base.:*(A::AbstractMatrix, u::SquareRootGaussian)
    return SquareRootGaussian(A * u.m, A * u.Z)
end

Statistics.mean(u::SquareRootGaussian) = u.m
Statistics.cov(u::SquareRootGaussian) = u.Z * u.Z'
Statistics.var(u::SquareRootGaussian) = sum(u.Z .^ 2, dims = 2)[:, 1]
Statistics.std(u::SquareRootGaussian) = sqrt.(Statistics.var(u))
