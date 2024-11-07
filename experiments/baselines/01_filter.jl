include("common.jl")


ys_train = jldopen(data_path, "r") do file
    file["ys_train"]
end


rng = Random.seed!(2345)


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
        uₖ = ComputationAwareKalmanExperiments.EnsembleKalmanFilter.update_enkf(
            u⁻ₖ,
            ys_train[k],
            H,
            Λ,
            rng,
        )
        push!(us, uₖ)
    end

    return us[2:end]
end
