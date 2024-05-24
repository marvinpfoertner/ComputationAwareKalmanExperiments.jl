#!/usr/bin/env julia

using GLMakie

config_id = length(ARGS) > 0 ? ARGS[1] : "debug"

include("../init_experiment.jl")

include("utils.jl")

with_ds(era5) do ds
    idx = Observable(1)

    fig = Figure()

    ax = axis_sphere(fig[1, 1]; title = "Data")

    p = plot_heatmap_sphere!(ax, era5, @lift(T₂ₘs(era5, ds, $idx)), colormap = :coolwarm)

    Colorbar(fig[1, 2], p)

    Makie.record(
        fig,
        "$(config.results_path)/data.mp4",
        1:length(era5.ts),
        framerate = 12,
    ) do k
        idx[] = k
    end

    fig
end
