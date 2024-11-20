function initialize_sample(
    gmc::ComputationAwareKalman.AbstractGaussMarkovChain;
    rng::Random.AbstractRNG,
    rank::Integer,
)
    E₀ = hcat([rand(rng, gmc, 0) for _ = 1:rank]...)
    return ensemble_to_gaussian(E₀)
end

function initialize_truncate(
    gmp::Union{
        ComputationAwareKalman.AbstractGaussMarkovProcess,
        ComputationAwareKalman.AbstractGaussMarkovChain,
    },
    t::Real = 0;
    rank::Integer,
    truncate_kwargs = (;),
)
    m₀ = ComputationAwareKalman.μ(gmp, t)
    Z₀, _ = ComputationAwareKalman.truncate(
        ComputationAwareKalman.lsqrt_Σ(gmp, t);
        max_cols = rank,
        truncate_kwargs...,
    )
    return SquareRootGaussian(m₀, Z₀)
end

function initialize_lanczos(
    gmc::Union{
        ComputationAwareKalman.AbstractGaussMarkovProcess,
        ComputationAwareKalman.AbstractGaussMarkovChain,
    },
    t::Real = 0;
    rng::Random.AbstractRNG,
    rank::Integer,
)
    m₀ = ComputationAwareKalman.μ(gmc, t)
    Σ₀ = ComputationAwareKalman.Σ(gmc, t)

    # Low-rank approximation of left square root of Σ₀
    eigvals, eigvecs, _ = KrylovKit.eigsolve(
        x -> Σ₀ * x,
        randn(rng, length(m₀)),
        rank,
        :LM;
        krylovdim = rank,
        maxiter = 1,
        orth = KrylovKit.ClassicalGramSchmidt2(),
        issymmetric = true,
    )
    Z₀ = hcat(eigvecs...) * Diagonal(sqrt.(eigvals))

    return SquareRootGaussian(m₀, Z₀)
end
