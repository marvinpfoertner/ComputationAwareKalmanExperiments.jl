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
    mmod::ComputationAwareKalman.AbstractMeasurementModel,
    ys,
    ts;
    rng::Random.AbstractRNG,
    rank::Integer,
)
    filter_states_data = EnsembleKalman.enkf(dgmp, mmod, ys; rng = rng, rank = rank)

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

function etkf(
    dgmp::ComputationAwareKalman.DiscretizedGaussMarkovProcess{
        <:ComputationAwareKalman.SpatiallyDiscretizedSTSGMP,
    },
    mmod::ComputationAwareKalman.AbstractMeasurementModel,
    ys,
    ts;
    rank::Integer,
    truncate_kwargs = (;),
)
    # Low-rank approximate the prior covariance matrix
    gmp_trunc = EnsembleKalman.discretize_truncate(dgmp.gmp.stsgmp, dgmp.gmp.X; rank = rank)
    dgmp_trunc = ComputationAwareKalman.discretize(gmp_trunc, dgmp.ts)

    # Filter
    filter_states_data = EnsembleKalman.etkf(
        dgmp_trunc,
        mmod,
        ys;
        rank = rank,
        truncate_kwargs = truncate_kwargs,
    )

    # Interpolate
    return [
        EnsembleKalman.interpolate(
            dgmp_trunc.gmp,
            ComputationAwareKalman.ts(dgmp_trunc),
            filter_states_data,
            t,
        ) for t in ts
    ]
end

function cakf(
    dgmp::ComputationAwareKalman.DiscretizedGaussMarkovProcess,
    mmod::ComputationAwareKalman.AbstractMeasurementModel,
    ys,
    ts;
    rank::Integer,
)
    fcache = ComputationAwareKalman.filter(
        dgmp,
        mmod,
        ys;
        update_kwargs = (max_iter = rank,),
        truncate_kwargs = (max_cols = rank,),
    )

    return [ComputationAwareKalman.interpolate(dgmp, fcache, t) for t in ts]
end
