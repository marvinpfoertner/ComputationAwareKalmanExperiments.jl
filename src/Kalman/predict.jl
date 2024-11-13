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