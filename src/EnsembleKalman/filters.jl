function enkf(
    dgmp::ComputationAwareKalman.AbstractGaussMarkovChain,
    mmod::ComputationAwareKalman.AbstractMeasurementModel,
    ys;
    rng::Random.AbstractRNG,
    rank::Integer,
)
    u₀ = initialize_sample(dgmp, rng = rng, rank = rank)

    states = typeof(u₀)[]
    uₖ₋₁ = u₀

    for (k, yₖ) in enumerate(ys)
        u⁻ₖ = predict_sample_pointwise(
            uₖ₋₁,
            ComputationAwareKalman.A_b_lsqrt_Q(dgmp, k - 1)...,
            rng,
        )

        if ismissing(yₖ)
            uₖ = u⁻ₖ
        else
            uₖ = update_enkf(
                u⁻ₖ,
                yₖ,
                ComputationAwareKalman.H(mmod, k),
                ComputationAwareKalman.Λ(mmod, k),
                rng,
            )
        end

        push!(states, uₖ)
        uₖ₋₁ = uₖ
    end

    return states
end

function etkf(
    dgmp::ComputationAwareKalman.AbstractGaussMarkovChain,
    mmod::ComputationAwareKalman.AbstractMeasurementModel,
    ys;
    rng::Random.AbstractRNG,
    rank::Integer,
)
    u₀ = initialize_sample(dgmp, rng = rng, rank = rank)

    states = typeof(u₀)[]
    uₖ₋₁ = u₀

    for (k, yₖ) in enumerate(ys)
        u⁻ₖ = predict_sample_pointwise(
            uₖ₋₁,
            ComputationAwareKalman.A_b_lsqrt_Q(dgmp, k - 1)...,
            rng,
        )

        if ismissing(yₖ)
            uₖ = u⁻ₖ
        else
            uₖ = update_etkf(
                u⁻ₖ,
                yₖ,
                ComputationAwareKalman.H(mmod, k),
                ComputationAwareKalman.Λ(mmod, k),
            )
        end

        push!(states, uₖ)
        uₖ₋₁ = uₖ
    end

    return states
end
