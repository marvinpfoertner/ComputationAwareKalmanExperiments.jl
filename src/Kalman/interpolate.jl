function interpolate(
    dgmp::ComputationAwareKalman.DiscretizedGaussMarkovProcess,
    uᶠs::AbstractVector{<:Gaussian},
    t::Real,
)
    k = searchsortedlast(ComputationAwareKalman.ts(dgmp), t)

    if k < 1
        uᶠₜ = Gaussian(
            ComputationAwareKalman.μ(dgmp.gmp, t),
            ComputationAwareKalman.Σ(dgmp.gmp, t),
        )
    elseif t == ComputationAwareKalman.ts(dgmp)[k]
        uᶠₜ = uᶠs[k]
    else
        uᶠₜ = predict(
            uᶠs[k],
            ComputationAwareKalmanExperiments.transition_model(
                dgmp.gmp,
                t,
                ComputationAwareKalman.ts(dgmp)[k],
            )...,
        )
    end

    return uᶠₜ
end