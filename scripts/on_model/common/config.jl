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
