module KalmanFilter

using LinearAlgebra
using Statistics

struct SquareRootGaussian{Tm<:AbstractVector,TZ<:AbstractMatrix}
    m::Tm
    Z::TZ
end

function Base.:*(A::AbstractMatrix, u::SquareRootGaussian)
    return SquareRootGaussian(A * u.m, A * u.Z)
end

function Statistics.var(u::SquareRootGaussian)
    return sum(u.Z .^ 2, dims = 2)[:, 1]
end

function predict(
    u::SquareRootGaussian,
    A::AbstractMatrix,
    b::AbstractVector,
    lsqrt_Q::AbstractMatrix,
)
    m⁻ = A * u.m + b
    Z⁻ = lq!([A * u.Z;; lsqrt_Q]).L
    return SquareRootGaussian(m⁻, Z⁻)
end

function update(
    u⁻::SquareRootGaussian,
    y::AbstractVector,
    H::AbstractMatrix,
    lsqrt_Λ::AbstractMatrix,
)
    S = Cholesky(LowerTriangular(lq!([H * u⁻.Z;; lsqrt_Λ]).L))
    K = u⁻.Z * (u⁻.Z' * (H' / S))

    m = u⁻.m + K * (y - H * u⁻.m)
    Z = lq!([u⁻.Z - K * (H * u⁻.Z);; K * lsqrt_Λ]).L

    return SquareRootGaussian(m, Z)
end

end
