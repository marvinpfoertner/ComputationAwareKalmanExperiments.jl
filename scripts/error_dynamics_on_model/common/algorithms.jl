function kf_rts(
    dgmp::ComputationAwareKalman.DiscretizedGaussMarkovProcess,
    mmod::ComputationAwareKalman.AbstractMeasurementModel,
    ys,
    ts,
)
    # Filter
    filter_states_data = Kalman.kf(dgmp, mmod, ys)
    filter_states = [Kalman.interpolate(dgmp, filter_states_data, t) for t in ts]

    # Smoother
    smoother_states_data = Kalman.rts(dgmp, filter_states_data)
    smoother_states =
        [Kalman.interpolate(dgmp, filter_states_data, smoother_states_data, t) for t in ts]

    return @ntuple(filter_states, smoother_states)
end

function cakf_caks(
    dgmp::ComputationAwareKalman.DiscretizedGaussMarkovProcess,
    dgmp_dev::ComputationAwareKalman.DiscretizedGaussMarkovProcess,
    mmod::ComputationAwareKalman.AbstractMeasurementModel,
    ys,
    ts;
    rank::Integer,
)
    # Filter
    fcache = ComputationAwareKalman.filter(
        dgmp_dev,
        mmod,
        ys;
        update_kwargs = (max_iter = rank,),
        truncate_kwargs = (max_cols = rank,),
    )
    filter_states = [ComputationAwareKalman.interpolate(dgmp, fcache, t) for t in ts]

    # Smoother
    scache = ComputationAwareKalman.smoother(
        dgmp_dev,
        fcache;
        truncate_kwargs = (max_cols = rank,),
    )

    smoother_states =
        [ComputationAwareKalman.interpolate(dgmp, fcache, scache, t) for t in ts]

    return @ntuple(filter_states, smoother_states)
end
