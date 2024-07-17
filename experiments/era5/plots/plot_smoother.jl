#!/usr/bin/env julia

using GLMakie
using Colors

config_id = length(ARGS) > 0 ? ARGS[1] : "debug"

include("../init_experiment.jl")

include("utils.jl")

fcache, data_mean = jldopen(filter_res_path, "r") do results
    fcache = results["cache"]
    fcache.path = cache_path

    return fcache, results["data_mean"]
end

scache = jldopen(smoother_res_path, "r") do results
    scache = results["cache"]
    scache.path = cache_path

    return scache
end

with_ds(era5) do ds
    xˢₖ = Observable(ComputationAwareKalman.interpolate(dgmp, fcache, scache, era5.ts[1]))

    fig = Figure(size = (4320, 2160))

    # Plot mean
    ax_mean = axis_sphere(fig[1, 1])

    plot_heatmap_sphere!(
        ax_mean,
        era5,
        @lift(H_plot * mean($xˢₖ) .+ data_mean),
        colormap = :coolwarm,
    )

    # Plot standard deviation
    ax_std = axis_sphere(fig[1, 2])

    p_std = plot_heatmap_sphere!(
        ax_std,
        era5,
        @lift(sqrt.(H_plot * diag(cov($xˢₖ)))),
        colormap = Colors.colormap("Purples"),
    )

    @withprogress name = "Animating..." begin
        Makie.record(
            fig,
            "$(config.results_path)/smoother.mp4",
            1:length(era5.ts),
            framerate = 12,
        ) do k
            xˢₖ[] = ComputationAwareKalman.interpolate(dgmp, fcache, scache, era5.ts[k])

            @logprogress k / length(era5.ts)
        end
    end

    fig
end
