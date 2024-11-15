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

function interpolate(
    dgmp::ComputationAwareKalman.DiscretizedGaussMarkovProcess,
    uᶠs::AbstractVector{<:SquareRootGaussian},
    t::Real,
)
    k = searchsortedlast(ComputationAwareKalman.ts(dgmp), t)

    if k < 1
        uᶠₜ = SquareRootGaussian(
            ComputationAwareKalman.μ(dgmp.gmp, t),
            ComputationAwareKalman.lsqrt_Σ(dgmp.gmp, t),
        )
    elseif t == ComputationAwareKalman.ts(dgmp)[k]
        uᶠₜ = uᶠs[k]
    else
        uᶠₜ = predict(
            uᶠs[k],
            ComputationAwareKalman.A_b_lsqrt_Q(
                dgmp.gmp,
                t,
                ComputationAwareKalman.ts(dgmp)[k],
            )...,
        )
    end

    return uᶠₜ
end
