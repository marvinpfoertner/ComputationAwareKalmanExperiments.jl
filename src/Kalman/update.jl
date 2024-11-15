function update(u⁻::Gaussian, y::AbstractVector, H::AbstractMatrix, Λ::AbstractMatrix)
    S = hermitianpart!(H * u⁻.P * H' + Λ)
    K = u⁻.P * (H' / S)

    m = u⁻.m + K * (y - H * u⁻.m)
    P = u⁻.P - K * H * u⁻.P

    return Gaussian(m, P)
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
