function filter(init_state::SquareRootGaussian, predict, update, ys)
    uᶠs = SquareRootGaussian[]

    uᶠₖ₋₁ = init_state

    for (k, yₖ) in enumerate(ys)
        u⁻ₖ = predict(uᶠₖ₋₁, k)

        if ismissing(yₖ)
            uᶠₖ = u⁻ₖ
        else
            uᶠₖ = update(u⁻ₖ, k, yₖ)
        end

        push!(uᶠs, uᶠₖ)
        uᶠₖ₋₁ = uᶠₖ
    end

    return uᶠs
end

function enkf(
    gmc::ComputationAwareKalman.AbstractGaussMarkovChain,
    mmod::ComputationAwareKalman.AbstractMeasurementModel,
    ys;
    rank::Integer,
    rng::Random.AbstractRNG,
)
    u₀ = initialize_sample(gmc; rng = rng, rank = rank)

    function predict(uᶠₖ₋₁, k)
        return predict_sample_pointwise(
            uᶠₖ₋₁,
            ComputationAwareKalman.A_b_lsqrt_Q(gmc, k - 1)...,
            rng,
        )
    end

    function update(u⁻ₖ, k, yₖ)
        return update_enkf(
            u⁻ₖ,
            yₖ,
            ComputationAwareKalman.H(mmod, k),
            ComputationAwareKalman.Λ(mmod, k),
            rng,
        )
    end

    return filter(u₀, predict, update, ys)
end

function etkf(
    gmc::ComputationAwareKalman.AbstractGaussMarkovChain,
    mmod::ComputationAwareKalman.AbstractMeasurementModel,
    ys;
    rank::Integer,
    rng::Random.AbstractRNG,
    lanczos_kwargs = (;),
    init_state::SquareRootGaussian = initialize_lanczos(
        gmc;
        rank = rank,
        initvec = randn(rng, ComputationAwareKalman.statedim(gmc)),
        lanczos_kwargs = lanczos_kwargs,
    ),
    predict = (uᶠₖ₋₁, k) -> predict_lanczos(
        uᶠₖ₋₁,
        ComputationAwareKalmanExperiments.transition_model(gmc, k - 1)...;
        rank = rank,
        initvec = randn(rng, size(uᶠₖ₋₁.m, 1)),
        lanczos_kwargs = lanczos_kwargs,
    ),
)
    function update(u⁻ₖ, k, yₖ)
        return update_etkf(
            u⁻ₖ,
            yₖ,
            ComputationAwareKalman.H(mmod, k),
            ComputationAwareKalman.Λ(mmod, k),
        )
    end

    return filter(init_state, predict, update, ys)
end
