function interpolate_truncate(
    dgmp::ComputationAwareKalman.DiscretizedGaussMarkovProcess,
    uᶠs::AbstractVector{<:SquareRootGaussian},
    t::Real;
    rank::Integer,
    truncate_kwargs = (;),
)
    k = searchsortedlast(ComputationAwareKalman.ts(dgmp), t)

    if k < 1
        uᶠₜ =
            initialize_truncate(dgmp.gmp, t; rank = rank, truncate_kwargs = truncate_kwargs)
    elseif t == ComputationAwareKalman.ts(dgmp)[k]
        uᶠₜ = uᶠs[k]
    else
        uᶠₜ = predict_truncate(
            uᶠs[k],
            ComputationAwareKalman.A_b_lsqrt_Q(
                dgmp.gmp,
                t,
                ComputationAwareKalman.ts(dgmp)[k],
            )...;
            rank = rank,
            truncate_kwargs = truncate_kwargs,
        )
    end

    return uᶠₜ
end
