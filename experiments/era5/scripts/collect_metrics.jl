#!/usr/bin/env julia

configs = [
    "24-1",
    "24-2",
    "24-4",
    "24-8",
    "24-16",
    "24-32",
    "24-64",
    "24-128",
    "24-256",
    "24-512",
    "12-1",
    "12-2",
    "12-4",
    "12-8",
    "12-16",
    "12-32",
    "12-64",
    "12-128",
    "12-256",
    "12-512",
    "6-1",
    "6-2",
    "6-4",
    "6-8",
    "6-16",
    "6-32",
    "6-64",
    "6-128",
    "6-256",
    "12-1-random",
    "12-2-random",
    "12-4-random",
    "12-8-random",
    "12-16-random",
    "12-32-random",
    "12-64-random",
    "12-128-random",
    "12-256-random",
    "12-512-random",
    "12-1-coordinate",
    "12-2-coordinate",
    "12-4-coordinate",
    "12-8-coordinate",
    "12-16-coordinate",
    "12-32-coordinate",
    "12-64-coordinate",
    "12-128-coordinate",
    "12-256-coordinate",
    "12-512-coordinate",
]

results_path = normpath(joinpath(@__DIR__, "../results"))

output_path = normpath(ARGS[1])
mkpath(output_path)

for config in configs
    metrics_file_path = joinpath(results_path, config, "metrics.jld2")

    if isfile(metrics_file_path)
        dst_path = joinpath(output_path, config)
        mkpath(dst_path)

        cp(metrics_file_path, joinpath(dst_path, "metrics.jld2"))
    end
end
