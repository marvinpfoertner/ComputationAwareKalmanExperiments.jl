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

function predict_truncate(
    u::SquareRootGaussian,
    A::AbstractMatrix,
    b::AbstractVector,
    lsqrt_Q::AbstractMatrix;
    rank::Integer = size(u.Z, 2),
    truncate_kwargs = (;),
)
    Z⁻, _ = ComputationAwareKalman.truncate(
        [A * u.Z;; lsqrt_Q];
        max_cols = rank,
        truncate_kwargs...,
    )

    return SquareRootGaussian(A * u.m + b, Z⁻)
end

function predict_lanczos(
    u::SquareRootGaussian,
    A::AbstractMatrix,
    b::AbstractVector,
    Q::AbstractMatrix;
    rng::Random.AbstractRNG,
    rank::Integer = size(u.Z, 2),
)
    AZ = A * u.Z

    # Low-rank approximation of left square root of predictive covariance
    initvec = randn(rng, size(AZ, 1))
    # initvec = mean(AZ, dims = 2)[:, 1]

    eigvals, eigvecs, _ = KrylovKit.eigsolve(
        x -> AZ * (AZ' * x) + Q * x,
        initvec,
        rank,
        :LM;
        krylovdim = max(KrylovDefaults.krylovdim, rank),
        orth = KrylovKit.ClassicalGramSchmidt2(),
        issymmetric = true,
    )

    Z⁻ = hcat(eigvecs...) * Diagonal(sqrt.(eigvals))

    return SquareRootGaussian(A * u.m + b, Z⁻)
end
