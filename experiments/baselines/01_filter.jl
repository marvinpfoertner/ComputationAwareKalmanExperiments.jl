include("common.jl")


ys_train = jldopen(data_path, "r") do file
    file["ys_train"]
end


rng = Random.seed!(2345)

ys_train_aug = Vector{Union{eltype(ys_train),Missing}}(missing, length(ts))
ys_train_aug[ts_train_idcs] = ys_train

function filter_enkf(dgmp, H, Λ, ys_train, rng, rank)
    E₀ = hcat([rand(rng, dgmp, 0) for _ = 1:rank]...)
    u₀ = ComputationAwareKalmanExperiments.EnsembleKalmanFilter.EnsembleGaussian(E₀)

    us = [u₀]

    for k in axes(ys_train, 1)
        u⁻ₖ =
            ComputationAwareKalmanExperiments.EnsembleKalmanFilter.predict_sample_pointwise(
                dgmp,
                k - 1,
                us[end],
                rng,
            )

        if ismissing(ys_train[k])
            uₖ = u⁻ₖ
        else
            uₖ = ComputationAwareKalmanExperiments.EnsembleKalmanFilter.update_enkf(
                u⁻ₖ,
                ys_train[k],
                H,
                Λ,
                rng,
            )
        end

        push!(us, uₖ)
    end

    return us[2:end]
end

function filter_etkf(dgmp, H, Λ, ys_train, rng, rank)
    E₀ = hcat([rand(rng, dgmp, 0) for _ = 1:rank]...)
    u₀ = ComputationAwareKalmanExperiments.EnsembleKalmanFilter.EnsembleGaussian(E₀)

    us = [u₀]

    for k in axes(ys_train, 1)
        u⁻ₖ =
            ComputationAwareKalmanExperiments.EnsembleKalmanFilter.predict_sample_pointwise(
                dgmp,
                k - 1,
                us[end],
                rng,
            )

        if ismissing(ys_train[k])
            uₖ = u⁻ₖ
        else
            uₖ = ComputationAwareKalmanExperiments.EnsembleKalmanFilter.update_etkf(
                u⁻ₖ,
                ys_train[k],
                H,
                Λ,
            )
        end

        push!(us, uₖ)
    end

    return us[2:end]
end
