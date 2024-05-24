#!/usr/bin/env julia

using GLMakie
using TuePlots
using Colors

config_id = length(ARGS) > 0 ? ARGS[1] : "debug"

include("../init_experiment.jl")

include("utils.jl")

T = Theme(
    TuePlots.SETTINGS[:NEURIPS];
    font = true,
    fontsize = true,
    single_column = false,
    figsize = true,
    thinned = true,
    subplot_height_to_width_ratio = 1.3,
    width_coeff = 0.25,
)

Tmin = -48.51
Tmax = 36.91
stdmin = 0.0
stdmax = 10.0

# Load values
n = length(dgmp)

yₙ = with_ds(era5) do ds
    T₂ₘs(era5, ds, era5_split.t_idcs_train[n])
end

mˢₙ, σˢₙ = jldopen(filter_res_path, "r") do results
    fcache = results["cache"]
    fcache.path = cache_path

    data_mean = results["data_mean"]

    mˢₙ = H_plot * (ComputationAwareKalman.m(fcache, n) .+ data_mean)
    σˢₙ = sqrt.(H_plot * diag(ComputationAwareKalman.P(dgmp, fcache, n)))

    return mˢₙ, σˢₙ
end

function temp_cmap_scale(relval)
    T = relval * (Tmax - Tmin) + Tmin

    return -T / (T >= 0.0 ? Tmax : abs(Tmin))
end

centered_coolwarm = cgrad(:coolwarm, scale = temp_cmap_scale)

plot_params = Dict(
    "$(n)_data" => (
        values = yₙ,
        color_kw = (colormap = centered_coolwarm, colorrange = (Tmin, Tmax)),
        ticks = [-45, 0, 35],
    ),
    "$(n)_smoother_mean" => (
        values = mˢₙ,
        color_kw = (colormap = centered_coolwarm, colorrange = (Tmin, Tmax)),
        ticks = [-45, 0, 35],
    ),
    "$(n)_smoother_std" => (
        values = σˢₙ,
        color_kw = (
            colormap = cgrad(Colors.colormap("Purples"), scale = exp),
            colorrange = (stdmin, stdmax),
        ),
        ticks = [0, 5, 10],
    ),
)

for plot_name in keys(plot_params)
    with_theme(T) do
        fig = Makie.Figure(figure_padding = 0)

        lscene = LScene(
            fig[1, 1],
            show_axis = false,
            scenekw = (projection = :Orthographic, clear = true, center = false),
        )

        scene = lscene.scene

        with_ds(era5) do ds
            plot_heatmap_sphere!(
                scene,
                era5,
                plot_params[plot_name].values;
                plot_params[plot_name].color_kw...,
            )
        end

        Makie.rotate!(Accum, scene, 14.0 * π / 180)
        center!(scene, 0)
        zoom!(scene, 0.55)

        Colorbar(
            fig[2, 1],
            vertical = false,
            flipaxis = false,
            height = 6,
            width = Relative(3 / 4),
            ticks = plot_params[plot_name].ticks;
            tickformat = "{:.0f} °C",
            tickwidth = 0.5,
            plot_params[plot_name].color_kw...,
        )

        # Box(fig[1, 1], color=:transparent, strokecolor=:red)

        rowgap!(fig.layout, 8)

        save(
            "$(config.results_path)/$plot_name.png",
            fig,
            px_per_unit = 30.0,
            update = false,
        )
    end
end
