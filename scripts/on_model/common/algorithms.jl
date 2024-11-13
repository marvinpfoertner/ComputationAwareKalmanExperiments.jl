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
