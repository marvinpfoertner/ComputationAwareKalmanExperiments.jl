function predict(u::Gaussian, A::AbstractMatrix, b::AbstractVector, Q::AbstractMatrix)
    return Gaussian(A * u.m + b, A * u.P * A' + Q)
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