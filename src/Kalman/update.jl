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
