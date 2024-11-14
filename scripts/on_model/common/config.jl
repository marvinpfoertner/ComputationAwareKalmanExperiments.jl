ranks = [4, 8, 16, 32, 64, 128]
seeds = 1:10

configs = Dict(
    "srkf" => Dict("algorithm" => "srkf"),
    "enkf" => Dict(
        rank => [
            Dict("algorithm" => "enkf", "rank" => rank, "seed" => seed) for seed in seeds
        ] for rank in ranks
    ),
    "etkf" => Dict(
        rank => [
            Dict("algorithm" => "etkf", "rank" => rank, "seed" => seed) for seed in seeds
        ] for rank in ranks
    ),
    "cakf" =>
        Dict(rank => Dict("algorithm" => "cakf", "rank" => rank) for rank in ranks),
)

#####################
# Common parameters #
#####################

# Problem
temporal_domain = (0.0, 5.0)
spatial_domain = (-5.0, 5.0)

# Dynamics
matern_order_t = 2
lₜ = 0.5
σ² = 1.0

lₓ = 1.5
Σₓ = Matern52Kernel() ∘ ScaleTransform(1.0 / lₓ)

# Discretization
Nₜ = 100
Nₓ = 1000

# Measurement
λ² = 0.1^2

# Data
data_seed = 2345

Nₜ_train = 10
Nₓ_train = 300
