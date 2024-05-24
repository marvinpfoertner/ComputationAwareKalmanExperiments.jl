const results_path = normpath(joinpath(@__DIR__, "..", "results"))

struct ERA5ExperimentConfig{Ttidcs}
    config_id::String
    t_idcs::Ttidcs
    step_λθ::Int
    step_λθ_test::Int
    lₜ::Float64
    lₓ::Float64
    σ²::Float64
    Λ::Float64
    policy::String
    abstol::Float64
    reltol::Float64
    max_iter::Int
    min_sval::Float64
    max_rank::Int
    results_path::String
end

function default_config(
    config_id::String,
    step_λθ::Int,
    max_iter::Int,
    max_rank::Int = 2 * max_iter;
    policy::String = "cg",
)
    t_idcs = 1:2*24

    step_λθ_test = 2

    lₜ = 3 / 24  # [days]
    lₓ = step_λθ * step_λθ_test * (C_earth / Nλ_total)  # [km]
    σ² = 10.0^2  # [(°C)²]

    Λ = 0.1^2  # [(°C)²]

    abstol = 1e-6
    reltol = eps(Float64)

    min_sval = sqrt(eps(Float64))

    return ERA5ExperimentConfig(
        config_id,
        t_idcs,
        step_λθ,
        step_λθ_test,
        lₜ,
        lₓ,
        σ²,
        Λ,
        policy,
        abstol,
        reltol,
        max_iter,
        min_sval,
        max_rank,
        joinpath(results_path, config_id),
    )
end

configs = [
    default_config("debug", 36, 64),
    #! format: off
    default_config("24-1",               24, 2^0),
    default_config("24-2",               24, 2^1),
    default_config("24-4",               24, 2^2),
    default_config("24-8",               24, 2^3),
    default_config("24-16",              24, 2^4),
    default_config("24-32",              24, 2^5),
    default_config("24-64",              24, 2^6),
    default_config("24-128",             24, 2^7),
    default_config("24-256",             24, 2^8),
    default_config("24-512",             24, 2^9),
    default_config("24-1024",            24, 2^10),
    default_config("12-1",               12, 2^0),
    default_config("12-2",               12, 2^1),
    default_config("12-4",               12, 2^2),
    default_config("12-8",               12, 2^3),
    default_config("12-16",              12, 2^4),
    default_config("12-32",              12, 2^5),
    default_config("12-64",              12, 2^6),
    default_config("12-128",             12, 2^7),
    default_config("12-256",             12, 2^8),
    default_config("12-512",             12, 2^9),
    default_config("6-1",                 6, 2^0),
    default_config("6-2",                 6, 2^1),
    default_config("6-4",                 6, 2^2),
    default_config("6-8",                 6, 2^3),
    default_config("6-16",                6, 2^4),
    default_config("6-32",                6, 2^5),
    default_config("6-64",                6, 2^6),
    default_config("6-128",               6, 2^7),
    default_config("6-256",               6, 2^8),
    default_config("3-1",                 3, 2^0),
    default_config("3-2",                 3, 2^1),
    default_config("3-4",                 3, 2^2),
    default_config("3-8",                 3, 2^3),
    default_config("3-16",                3, 2^4),
    default_config("3-32",                3, 2^5),
    default_config("3-64",                3, 2^6),
    default_config("12-1-random",        12, 2^0, policy="random"),
    default_config("12-2-random",        12, 2^1, policy="random"),
    default_config("12-4-random",        12, 2^2, policy="random"),
    default_config("12-8-random",        12, 2^3, policy="random"),
    default_config("12-16-random",       12, 2^4, policy="random"),
    default_config("12-32-random",       12, 2^5, policy="random"),
    default_config("12-64-random",       12, 2^6, policy="random"),
    default_config("12-128-random",      12, 2^7, policy="random"),
    default_config("12-256-random",      12, 2^8, policy="random"),
    default_config("12-512-random",      12, 2^9, policy="random"),
    default_config("12-1-coordinate",    12, 2^0, policy="coordinate"),
    default_config("12-2-coordinate",    12, 2^1, policy="coordinate"),
    default_config("12-4-coordinate",    12, 2^2, policy="coordinate"),
    default_config("12-8-coordinate",    12, 2^3, policy="coordinate"),
    default_config("12-16-coordinate",   12, 2^4, policy="coordinate"),
    default_config("12-32-coordinate",   12, 2^5, policy="coordinate"),
    default_config("12-64-coordinate",   12, 2^6, policy="coordinate"),
    default_config("12-128-coordinate",  12, 2^7, policy="coordinate"),
    default_config("12-256-coordinate",  12, 2^8, policy="coordinate"),
    default_config("12-512-coordinate",  12, 2^9, policy="coordinate"),
    #! format: on
]
configs =
    Dict{String,ERA5ExperimentConfig}(config.config_id => config for config in configs)
