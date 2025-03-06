function rts(
    dgmp::ComputationAwareKalman.AbstractGaussMarkovChain,
    uᶠs::AbstractVector{<:Gaussian},
)
    uˢₖ₊₁ = uᶠs[end]

    uˢs = [uˢₖ₊₁]

    for (k, uᶠₖ) in reverse(collect(enumerate(uᶠs[1:end-1])))
        u⁻ₖ₊₁ = predict(uᶠₖ, ComputationAwareKalmanExperiments.transition_model(dgmp, k)...)

        Kˢₖ = cov(uᶠₖ) * (ComputationAwareKalman.A(dgmp, k)' / cov(u⁻ₖ₊₁))

        mˢₖ = mean(uᶠₖ) + Kˢₖ * (mean(uˢₖ₊₁) - mean(u⁻ₖ₊₁))
        Pˢₖ = cov(uᶠₖ) + Kˢₖ * (cov(uˢₖ₊₁) - cov(u⁻ₖ₊₁)) * Kˢₖ'

        uˢₖ = Gaussian(mˢₖ, Pˢₖ)

        push!(uˢs, uˢₖ)
        uˢₖ₊₁ = uˢₖ
    end

    return reverse(uˢs)
end
