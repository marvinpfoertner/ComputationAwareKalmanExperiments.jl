sample_ranks = 2 .^ (1:10)
etkf_lanczos_ranks = 2 .^ (0:9)
cakf_ranks = 2 .^ (0:9)

seeds = 1:5

configs = (
    kf = [Dict("algorithm" => "kf")],
    srkf = [Dict("algorithm" => "srkf")],
    enkf = [
        Dict("algorithm" => "enkf", "rank" => rank, "seed" => seed) for
        rank in sample_ranks, seed in seeds
    ],
    etkf_sample = [
        Dict("algorithm" => "etkf-sample", "rank" => rank, "seed" => seed) for
        rank in sample_ranks, seed in seeds
    ],
    etkf_lanczos = [
        Dict("algorithm" => "etkf-lanczos", "rank" => rank, "seed" => seed) for
        rank in etkf_lanczos_ranks, seed in seeds
    ],
    cakf = [Dict("algorithm" => "cakf", "rank" => rank) for rank in cakf_ranks],
)

#####################
# Common parameters #
#####################

# Problem
temporal_domain = (0.0, 5.0)
spatial_domain = ((-10.0, 10.0), (-10.0, 10.0))

# Dynamics
matern_order_t = 1
lₜ = 0.5
σ² = 1.0

lₓ = 0.5

# Discretization
Nₜ = 100
Nₓ = (100, 100)

# Measurement
λ² = 0.1^2

# Data
data_seed = 2345

Nₜ_train = 10
Nₓ_train = 400
