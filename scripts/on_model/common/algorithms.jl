function kf(
    dgmp::ComputationAwareKalman.DiscretizedGaussMarkovProcess,
    mmod::ComputationAwareKalman.AbstractMeasurementModel,
    ys,
    ts,
)
    filter_states_data = Kalman.kf(dgmp, mmod, ys)

    return [Kalman.interpolate(dgmp, filter_states_data, t) for t in ts]
end

function srkf(
    dgmp::ComputationAwareKalman.DiscretizedGaussMarkovProcess,
    mmod::ComputationAwareKalman.AbstractMeasurementModel,
    ys,
    ts,
)
    filter_states_data = Kalman.srkf(dgmp, mmod, ys)

    return [Kalman.interpolate(dgmp, filter_states_data, t) for t in ts]
end

function enkf(
    dgmp::ComputationAwareKalman.DiscretizedGaussMarkovProcess,
    dgmp_dev::ComputationAwareKalman.DiscretizedGaussMarkovProcess,
    mmod::ComputationAwareKalman.AbstractMeasurementModel,
    ys,
    ts;
    rng::Random.AbstractRNG,
    rank::Integer,
)
    filter_states_data = EnsembleKalman.enkf(dgmp_dev, mmod, ys; rng = rng, rank = rank)

    # Interpolate
    return [
        EnsembleKalman.interpolate(
            dgmp.gmp,
            ComputationAwareKalman.ts(dgmp),
            filter_states_data,
            t,
        ) for t in ts
    ]
end

function etkf_sample(
    dgmp::ComputationAwareKalman.DiscretizedGaussMarkovProcess,
    mmod::ComputationAwareKalman.AbstractMeasurementModel,
    ys,
    ts;
    rng::Random.AbstractRNG,
    rank::Integer,
)
    filter_states_data = EnsembleKalman.etkf(
        dgmp,
        mmod,
        ys;
        rank = rank,
        rng = rng,
        init_state = EnsembleKalman.initialize_sample(dgmp, rank = rank, rng = rng),
        predict = (uᶠₖ₋₁, k) -> EnsembleKalman.predict_sample_pointwise(
            uᶠₖ₋₁,
            ComputationAwareKalman.A_b_lsqrt_Q(dgmp, k - 1)...,
            rng,
        ),
    )

    return [
        EnsembleKalman.interpolate(
            dgmp.gmp,
            ComputationAwareKalman.ts(dgmp),
            filter_states_data,
            t,
        ) for t in ts
    ]
end

function etkf_lanczos(
    dgmp::ComputationAwareKalman.DiscretizedGaussMarkovProcess,
    mmod::ComputationAwareKalman.AbstractMeasurementModel,
    ys,
    ts;
    rng::Random.AbstractRNG,
    rank::Integer,
)
    filter_states_data = EnsembleKalman.etkf(
        dgmp,
        mmod,
        ys;
        rank = rank,
        rng = rng,
        lanczos_kwargs = (krylovdim = rank,),
    )

    return [
        EnsembleKalman.interpolate(
            dgmp.gmp,
            ComputationAwareKalman.ts(dgmp),
            filter_states_data,
            t,
        ) for t in ts
    ]
end

function cakf(
    dgmp::ComputationAwareKalman.DiscretizedGaussMarkovProcess,
    dgmp_dev::ComputationAwareKalman.DiscretizedGaussMarkovProcess,
    mmod::ComputationAwareKalman.AbstractMeasurementModel,
    ys,
    ts;
    rank::Integer,
)
    fcache = ComputationAwareKalman.filter(
        dgmp_dev,
        mmod,
        ys;
        update_kwargs = (max_iter = rank,),
        truncate_kwargs = (max_cols = rank,),
    )

    return [ComputationAwareKalman.interpolate(dgmp, fcache, t) for t in ts]
end
