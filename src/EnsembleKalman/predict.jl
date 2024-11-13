function predict_sample_pointwise(
    u::SquareRootGaussian,
    A::AbstractMatrix,
    b::AbstractVector,
    lsqrt_Q::AbstractMatrix,
    rng::Random.AbstractRNG,
)
    # Low-rank approximation of left square root of Q by sampling
    qs = lsqrt_Q * randn(rng, (size(lsqrt_Q, 2), size(u.Z, 2)))
    Z_Q = (qs .- mean(qs, dims = 2)[:, 1]) / sqrt(size(qs, 2) - 1)

    return SquareRootGaussian(A * u.m + b, A * u.Z + Z_Q)
end
