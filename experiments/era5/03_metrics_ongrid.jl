#!/usr/bin/env julia

config_id = length(ARGS) > 0 ? ARGS[1] : "debug"

include("init_experiment.jl")

metrics = Dict{String,Any}(
    "state_space_dim" => size(H_plot, 2),
)

# Load filter results
stages = ["filter"]

fcache, data_mean = jldopen(filter_res_path, "r") do results
    metrics["filter/wall_time_ns"] = results["wall_time_ns"]

    fcache = results["cache"]
    fcache.path = cache_path

    return fcache, results["data_mean"]
end

# Load smoother results
if isfile(smoother_res_path)
    push!(stages, "smoother")

    scache = jldopen(smoother_res_path, "r") do results
        metrics["smoother/wall_time_ns"] = results["wall_time_ns"]

        scache = results["cache"]
        scache.path = cache_path

        return scache
    end
end

for stage in stages
    mses_train = Float64[]
    mses_test = Float64[]

    nlls_train = Float64[]
    nlls_test = Float64[]

    with_ds(era5) do ds
        @progress name = stage for k in 1:length(dgmp)
            yₖ = vec(T₂ₘs(era5, ds, era5_split.t_idcs_train[k])) .- data_mean

            if stage == "filter"
                μₖ = H_plot * ComputationAwareKalman.m(fcache, k)
                σ²ₖ = H_plot * diag(ComputationAwareKalman.P(dgmp, fcache, k))
            else
                @assert stage == "smoother"

                μₖ = H_plot * ComputationAwareKalman.mˢ(scache, k)
                σ²ₖ = H_plot * diag(ComputationAwareKalman.Pˢ(dgmp, scache, k))
            end

            # MSE
            sqerrsₖ = (yₖ - μₖ) .^ 2

            mseₖ_train = mean(sqerrsₖ[era5_split.λθ_idcs_train])
            mseₖ_test = mean(sqerrsₖ[era5_split.λθ_idcs_test])

            push!(mses_train, mseₖ_train)
            push!(mses_test, mseₖ_test)

            # (Normalized) NLL
            nllsₖ = gaussian_nll.(yₖ, μₖ, σ²ₖ)

            nllₖ_train = mean(nllsₖ[era5_split.λθ_idcs_train])
            nllₖ_test = mean(nllsₖ[era5_split.λθ_idcs_test])

            push!(nlls_train, nllₖ_train)
            push!(nlls_test, nllₖ_test)
        end
    end

    mean_mse_train = mean(mses_train)
    mean_mse_test = mean(mses_test)
    mean_nll_train = mean(nlls_train)
    mean_nll_test = mean(nlls_test)

    metrics["$stage/mse_train"] = mean_mse_train
    metrics["$stage/mse_test"] = mean_mse_test
    metrics["$stage/nll_train"] = mean_nll_train
    metrics["$stage/nll_test"] = mean_nll_test

    @show mean_mse_train
    @show mean_mse_test
    @show mean_nll_train
    @show mean_nll_test
end

save(metrics_path, metrics)
