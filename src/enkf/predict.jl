function predict_sample_pointwise(
    gmc::ComputationAwareKalman.AbstractGaussMarkovChain,
    k::Integer,
    uₖ₋₁::EnsembleGaussian,
    rng::Random.AbstractRNG,
)
    Eₖ₋₁ = members(uₖ₋₁)
    Eₖ = hcat([rand(rng, gmc, k, Eₖ₋₁[:, i]) for i in axes(Eₖ₋₁, 2)]...)

    return EnsembleGaussian(Eₖ)
end