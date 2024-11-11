using DrWatson

@quickactivate "ComputationAwareKalmanExperiments"

include("common.jl")

rng = Random.seed!(seed + 1)

ys_train_aug = Vector{Union{eltype(ys_train),Missing}}(missing, length(ts))
ys_train_aug[ts_train_idcs] = ys_train

function filter_enkf(dgmp, mmod, ys_train, rng, rank)
    E₀ = hcat([rand(rng, dgmp, 0) for _ = 1:rank]...)
    u₀ = ComputationAwareKalmanExperiments.EnsembleKalmanFilter.EnsembleGaussian(E₀)

    us = [u₀]

    for k in axes(ys_train, 1)
        u⁻ₖ =
            ComputationAwareKalmanExperiments.EnsembleKalmanFilter.predict_sample_pointwise(
                us[end],
                ComputationAwareKalman.A_b_lsqrt_Q(dgmp, k - 1)...,
                rng,
            )

        if ismissing(ys_train[k])
            uₖ = u⁻ₖ
        else
            uₖ = ComputationAwareKalmanExperiments.EnsembleKalmanFilter.update_enkf(
                u⁻ₖ,
                ys_train[k],
                ComputationAwareKalman.H(mmod, k),
                ComputationAwareKalman.Λ(mmod, k),
                rng,
            )
        end

        push!(us, uₖ)
    end

    return us[2:end]
end

function filter_etkf(dgmp, mmod, ys_train, rng, rank)
    E₀ = hcat([rand(rng, dgmp, 0) for _ = 1:rank]...)
    u₀ = ComputationAwareKalmanExperiments.EnsembleKalmanFilter.EnsembleGaussian(E₀)

    us = [u₀]

    for k in axes(ys_train, 1)
        u⁻ₖ =
            ComputationAwareKalmanExperiments.EnsembleKalmanFilter.predict_sample_pointwise(
                us[end],
                ComputationAwareKalman.A_b_lsqrt_Q(dgmp, k - 1)...,
                rng,
            )

        if ismissing(ys_train[k])
            uₖ = u⁻ₖ
        else
            uₖ = ComputationAwareKalmanExperiments.EnsembleKalmanFilter.update_etkf(
                u⁻ₖ,
                ys_train[k],
                ComputationAwareKalman.H(mmod, k),
                ComputationAwareKalman.Λ(mmod, k),
            )
        end

        push!(us, uₖ)
    end

    return us[2:end]
end
