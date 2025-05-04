include("common.jl")

# Load configuration
config = configs[config_id]

@show config_id
@show config

# Generate results path
mkpath(config.results_path)

# Load and downsample data
data_path = joinpath(@__DIR__, "data", "era5_t2m_2022.nc")

era5 = ERA5(
    data_path;
    t_idcs = config.t_idcs,
    step_λ = config.step_λθ,
    step_θ = config.step_λθ,
)

# Split data in train and test set
era5_split = ERA5TrainTestSplit(era5, Int[], config.step_λθ_test, config.step_λθ_test)

# Construct prior
stsgmp = prior(
    config.lₜ,  # [days]
    config.lₓ,  # [km]
    config.σ²,  # [(°C)²]
)

gmp, H_plot = discretize_space(stsgmp, era5.λθs)

# Measurement Model
dgmp = ComputationAwareKalman.discretize(gmp, ts_train(era5_split))

mmod = measurement_model(stsgmp.tgmp.H, era5_split, config.Λ);

# Result serialization
cache_path = joinpath(config.results_path, "caches")

filter_res_path = joinpath(config.results_path, "filter.jld2")
smoother_res_path = joinpath(config.results_path, "smoother.jld2")

metrics_path = joinpath(config.results_path, "metrics.jld2")

filter_log_path = joinpath(config.results_path, "filter.log")

function clear_results()
    if isdir(cache_path)
        rm(cache_path; recursive = true)
    end

    if isfile(filter_res_path)
        rm(filter_res_path)
    end

    if isfile(smoother_res_path)
        rm(smoother_res_path)
    end

    if isfile(metrics_path)
        rm(metrics_path)
    end

    if isfile(filter_log_path)
        rm(filter_log_path)
    end
end
