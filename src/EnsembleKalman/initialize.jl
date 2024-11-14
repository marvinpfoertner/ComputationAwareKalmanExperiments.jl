function initialize_sample(
    gmc::ComputationAwareKalman.AbstractGaussMarkovChain;
    rng::Random.AbstractRNG,
    rank::Integer,
)
    E₀ = hcat([rand(rng, gmc, 0) for _ = 1:rank]...)
    return ensemble_to_gaussian(E₀)
end
