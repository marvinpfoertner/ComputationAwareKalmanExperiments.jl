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
    uᶠs::AbstractVector{<:Gaussian},
    uˢs::AbstractVector{<:Gaussian},
    t::Real,
)
    k = searchsortedlast(ComputationAwareKalman.ts(dgmp), t)

    if k > 0 && t == ComputationAwareKalman.ts(dgmp)[k]
        return uˢs[k]
    end

    uᶠₜ = interpolate(dgmp, uᶠs, t)

    if k >= length(dgmp)
        return uᶠₜ
    end

    uˢₖ₊₁ = uˢs[k+1]
    u⁻ₖ₊₁ = predict(
        uᶠₜ,
        ComputationAwareKalmanExperiments.transition_model(
            dgmp.gmp,
            ComputationAwareKalman.ts(dgmp)[k+1],
            t,
        )...,
    )

    A₍ₖ₊₁₎ₜ = ComputationAwareKalman.A(dgmp.gmp, ComputationAwareKalman.ts(dgmp)[k+1], t)
    Kˢₜ = cov(uᶠₜ) * (A₍ₖ₊₁₎ₜ' / cov(u⁻ₖ₊₁))

    mˢₜ = mean(uᶠₜ) + Kˢₜ * (mean(uˢₖ₊₁) - mean(u⁻ₖ₊₁))
    Pˢₜ = cov(uᶠₜ) + Kˢₜ * (cov(uˢₖ₊₁) - cov(u⁻ₖ₊₁)) * Kˢₜ'

    return Gaussian(mˢₜ, Pˢₜ)
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
