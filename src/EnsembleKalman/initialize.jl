function initialize_sample(
    gmc::ComputationAwareKalman.AbstractGaussMarkovChain;
    rng::Random.AbstractRNG,
    rank::Integer,
)
    E₀ = hcat([rand(rng, gmc, 0) for _ = 1:rank]...)
    return ensemble_to_gaussian(E₀)
end

function initialize_lanczos(
    gmp::Union{
        ComputationAwareKalman.AbstractGaussMarkovProcess,
        ComputationAwareKalman.AbstractGaussMarkovChain,
    },
    t::Real = 0;
    rank::Integer,
    initvec::AbstractVector,
    lanczos_kwargs = (;),
)
    Σₜ = ComputationAwareKalman.Σ(gmp, t)

    eigvals, eigvecs, _ = KrylovKit.eigsolve(
        x -> Σₜ * x,
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

    Zₜ = hcat(eigvecs...) * Diagonal(sqrt.(max.(0.0, eigvals)))

    return SquareRootGaussian(ComputationAwareKalman.μ(gmp, t), Zₜ)
end
