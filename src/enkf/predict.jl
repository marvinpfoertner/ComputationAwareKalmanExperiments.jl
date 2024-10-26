function predict_sample_pointwise(
    gmc::ComputationAwareKalman.AbstractGaussMarkovChain,
    k::Integer,
    uₖ₋₁::EnsembleGaussian,
    rng::Random.AbstractRNG,
)
    Sₖ₋₁ = members(uₖ₋₁)
    Sₖ = hcat([rand(rng, gmc, k, Sₖ₋₁[:, i]) for i in axes(Sₖ₋₁, 2)]...)

    mₖ = sum(Sₖ, dims = 2) / size(Sₖ, 2)
    Zₖ = (Sₖ .- mₖ) / sqrt(size(Sₖ, 2) - 1)

    return EnsembleGaussian(mₖ, Zₖ)
end