#!/usr/bin/env julia

using Random

config_id = length(ARGS) > 0 ? ARGS[1] : "debug"

include("init_experiment.jl")

clear_results()

@show n = length(era5.λθs)
@show d = size(H_plot, 2)
@show n_train = length(era5_split.λθ_idcs_train)
@show n_test = length(era5_split.λθ_idcs_test)
@show n_train_all = length(dgmp) * length(era5_split.λθ_idcs_train)
@show n_test_all = length(dgmp) * length(era5_split.λθ_idcs_test)

# Set up logging
Logging.global_logger(
    TeeLogger(
        Logging.global_logger(),
        MinLevelLogger(
            FileLogger(filter_log_path; always_flush = true, append = true),
            Logging.Info,
        ),
    ),
)

y_mean = T₂ₘ_train_mean(era5_split)

fcache, filter_wall_ns = with_ds(era5) do ds
    fcache = ComputationAwareKalman.JLD2FilterCache(cache_path)

    mₖ₋₁ = ComputationAwareKalman.μ(dgmp, 0)
    M⁺ₖ₋₁ = zeros(Float64, size(mₖ₋₁, 1), 0)

    if config.policy == "random" || config.policy == "cg-random-1-1"
        rng = Random.seed!(425786)
    end

    tic = Base.time_ns()

    @progress name = "Filtering..." for k = 1:length(dgmp)
        # Predict
        m⁻ₖ, M⁻ₖ = ComputationAwareKalman.predict(dgmp, k, mₖ₋₁, M⁺ₖ₋₁)

        # Update
        yₖ = T₂ₘs_train(era5_split, ds, k) .- y_mean

        if config.policy == "cg"
            policy = ComputationAwareKalman.CGPolicy()
        elseif config.policy == "random"
            policy = ComputationAwareKalman.RandomGaussianPolicy(rng)
        elseif config.policy == "cg-random-1-1"
            policy = ComputationAwareKalman.MixedCGRandomGaussianPolicy(rng)
        else
            @assert config.policy == "coordinate"
            @assert config.max_iter <= length(yₖ)

            policy = ComputationAwareKalman.CoordinatePolicy(
                round.(Int, collect(range(1, length(yₖ), config.max_iter + 1))),
            )
        end

        tol = max(config.abstol, config.reltol * norm(yₖ, 2))

        function update_callback(; i, r, η, kwargs...)
            @info "Update Callback" k i norm(r, 2) η tol
        end

        xₖ = ComputationAwareKalman.update(
            m⁻ₖ,
            ComputationAwareKalman.Σ(dgmp, k),
            M⁻ₖ,
            ComputationAwareKalman.H(mmod, k),
            ComputationAwareKalman.Λ(mmod, k),
            yₖ,
            abstol = config.abstol,
            reltol = config.reltol,
            max_iter = config.max_iter,
            policy = policy,
            callback_fn = update_callback,
        )

        # Truncate
        M⁺ₖ, Π⁺ₖ = ComputationAwareKalman.truncate(
            xₖ.M;
            min_sval = config.min_sval,
            max_cols = config.max_rank,
        )

        push!(fcache; m⁻ = m⁻ₖ, xₖ..., M⁺ = M⁺ₖ, Π⁺ = Π⁺ₖ)

        mₖ₋₁ = xₖ.m
        M⁺ₖ₋₁ = M⁺ₖ
    end

    filter_wall_ns = Base.time_ns() - tic

    return fcache, filter_wall_ns
end

jldsave(filter_res_path; cache = fcache, wall_time_ns = filter_wall_ns, data_mean = y_mean)
