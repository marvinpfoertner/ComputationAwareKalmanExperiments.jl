function kf(
    gmc::ComputationAwareKalman.AbstractGaussMarkovChain,
    mmod::ComputationAwareKalman.AbstractMeasurementModel,
    ys,
    ts,
)
    uᶠs_train = Kalman.kf(gmc, mmod, ys)

    return [Kalman.interpolate(gmc, uᶠs_train, t) for t in ts]
end

function cakf(
    dgmp::ComputationAwareKalman.AbstractGaussMarkovChain,
    mmod::ComputationAwareKalman.AbstractMeasurementModel,
    ys;
    rank::Integer,
    ts,
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
    uᶠs = EnsembleKalman.etkf(
        dgmp_trunc,
        mmod,
        ys;
        rank = rank,
        truncate_kwargs = truncate_kwargs,
    )

    # Interpolate
    return [
        EnsembleKalman.interpolate_truncate(
            dgmp_trunc,
            uᶠs,
            t;
            rank = rank,
            truncate_kwargs = truncate_kwargs,
        ) for t in ts
    ]
end
