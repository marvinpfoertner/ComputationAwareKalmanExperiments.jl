ranks = [2, 3, 4, 5, 6, 7, 8, 9, 10]
seed = 345698

configs = Dict(
    "srkf" => Dict("algorithm" => "srkf"),
    "enkf" => Dict(
        rank => Dict("algorithm" => "enkf", "rank" => rank, "seed" => seed) for
        rank in ranks
    ),
    "etkf" => Dict(
        rank => Dict("algorithm" => "etkf", "rank" => rank, "seed" => seed) for
        rank in ranks
    ),
    "cakf" =>
        Dict(rank => Dict("algorithm" => "cakf", "rank" => rank) for rank in ranks),
)

# Problem parameters
temporal_domain = (0.0, 5.0)
spatial_domain = (-2.0, 2.0)

# Dynamics parameters
matern_order_t = 2
lₜ = 0.5
σ² = 1.0

lₓ = 1.5
Σₓ = Matern52Kernel() ∘ ScaleTransform(1.0 / lₓ)

# Discretization
Nₜ = 100
Nₓ = 200

# Measurement parameters
λ² = 0.1^2

# Data parameters
data_seed = 2345

Nₜ_train = 10
Nₓ_train = 20
