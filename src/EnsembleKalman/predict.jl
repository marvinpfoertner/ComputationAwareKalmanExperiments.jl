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

function predict_lanczos(
    u::SquareRootGaussian,
    A::AbstractMatrix,
    b::AbstractVector,
    Q::AbstractMatrix;
    rank::Integer = size(u.Z, 2),
    AZ = A * u.Z,
    initvec = mean(AZ, dims = 2)[:, 1],
    lanczos_kwargs = (;),
)
    eigvals, eigvecs, _ = KrylovKit.eigsolve(
        x -> AZ * (AZ' * x) + Q * x,
        initvec,
        rank,
        :LR,
        KrylovKit.Lanczos(;
            merge(
                (
                    krylovdim = rank,
                    maxiter = 1,
                    tol = 1e-10,
                    orth = KrylovKit.ModifiedGramSchmidt2,
                    eager = false,
                    verbosity = 0,
                ),
                lanczos_kwargs,
            )...,
        );
    )

    Z⁻ = hcat(eigvecs...) * Diagonal(sqrt.(max.(0.0, eigvals)))

    return SquareRootGaussian(A * u.m + b, Z⁻)
end
