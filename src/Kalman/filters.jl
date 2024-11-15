function kf(
    dgmp::ComputationAwareKalman.AbstractGaussMarkovChain,
    mmod::ComputationAwareKalman.AbstractMeasurementModel,
    ys,
)
    u₀ = Gaussian(ComputationAwareKalman.μ(dgmp, 0), ComputationAwareKalman.Σ(dgmp, 0))

    states = Gaussian[]
    uₖ₋₁ = u₀

    for (k, yₖ) in enumerate(ys)
        u⁻ₖ = predict(
            uₖ₋₁,
            ComputationAwareKalmanExperiments.transition_model(dgmp, k - 1)...,
        )

        uₖ = update(
            u⁻ₖ,
            yₖ,
            ComputationAwareKalman.H(mmod, k),
            ComputationAwareKalman.Λ(mmod, k),
        )

        push!(states, uₖ)
        uₖ₋₁ = uₖ
    end

    return states
end

function srkf(
    dgmp::ComputationAwareKalman.AbstractGaussMarkovChain,
    mmod::ComputationAwareKalman.AbstractMeasurementModel,
    ys,
)
    # Initialization
    u₀ = SquareRootGaussian(
        ComputationAwareKalman.μ(dgmp, 0),
        ComputationAwareKalman.lsqrt_Σ(dgmp, 0),
    )

    # Filter loop
    states = SquareRootGaussian[]
    uₖ₋₁ = u₀

    for (k, yₖ) in enumerate(ys)
        u⁻ₖ = predict(uₖ₋₁, ComputationAwareKalman.A_b_lsqrt_Q(dgmp, k - 1)...)

        if ismissing(yₖ)
            uₖ = u⁻ₖ
        else
            uₖ = update(
                u⁻ₖ,
                yₖ,
                ComputationAwareKalman.H(mmod, k),
                ComputationAwareKalman.lsqrt_Λ(mmod, k),
            )
        end

        push!(states, uₖ)
        uₖ₋₁ = uₖ
    end

    return states
end
