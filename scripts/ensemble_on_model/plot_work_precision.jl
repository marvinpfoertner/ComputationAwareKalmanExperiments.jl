using DrWatson
@quickactivate "ComputationAwareKalmanExperiments"

include("common.jl")

run_all()

# Collect metrics
using DataFrames
using Statistics

df = collect_results(datadir("ensemble_on_model"), rinclude = [r"metrics_algorithm="])

function collect_metrics(algorithm)
    df_algo = dropmissing(
        df[df.algorithm.==algorithm, [:seed, :rank, :wall_time, :mse, :expected_nll]],
    )
    df_metric_stats = combine(
        groupby(df_algo, :rank),
        :wall_time => Ref => :wall_times,
        :wall_time => median,
        :mse => Ref => :mses,
        :mse => median,
        :expected_nll => Ref => :expected_nlls,
        :expected_nll => median,
        :seed => Ref => :seeds,
    )

    return sort!(df_metric_stats, :rank)
end

metrics = (
    enkf = collect_metrics(:enkf),
    etkf_sample = collect_metrics(:etkf_sample),
    etkf_lanczos = collect_metrics(:etkf_lanczos),
    cakf = collect_metrics(:cakf),
)

# Generate plot
using CairoMakie
using TuePlots

CairoMakie.activate!(type = "svg")

labels = (enkf = "EnKF", etkf_sample = "ETKF-S", etkf_lanczos = "ETKF-L", cakf = "CAKF")
colors = (enkf = :orange, etkf_sample = :darkgray, etkf_lanczos = :black, cakf = :red)

function scatter_metric!(ax, xs, ys; markersize = 5, alpha = 0.5, kwargs...)
    scatter!(
        ax,
        vcat([(ndims(x) == 0) ? repeat([x], length(y)) : x for (x, y) in zip(xs, ys)]...),
        vcat(ys...);
        markersize = markersize,
        alpha = alpha,
        kwargs...,
    )
end

function work_precision_plot(;
    algorithms = [:enkf, :etkf_sample, :etkf_lanczos, :cakf],
    abscissa = :wall_time,
    xlabel = "Wall Time [s]",
    xscale = log10,
    mse_scale = identity,
    nll_scale = identity,
    legend_position = :lt,
)
    T = Theme(
        TuePlots.SETTINGS[:AISTATS];
        font = true,
        fontsize = true,
        single_column = true,
        figsize = true,
        thinned = true,
        nrows = 2,
        ncols = 1,
    )

    with_theme(T) do
        fig = Figure()

        axes = (
            mse = Axis(
                fig[1, 1],
                xlabel = xlabel,
                xscale = xscale,
                ylabel = "MSE",
                yscale = mse_scale,
            ),
            nll = Axis(
                fig[2, 1],
                xlabel = xlabel,
                xscale = xscale,
                ylabel = "Average NLPD",
                yscale = nll_scale,
            ),
        )

        for algorithm in algorithms
            if abscissa == :wall_time
                scatter_xs = metrics[algorithm].wall_times
                line_xs = metrics[algorithm].wall_time_median
            else
                scatter_xs = metrics[algorithm][:, abscissa]
                line_xs = metrics[algorithm][:, abscissa]
            end

            scatter_metric!(
                axes.mse,
                scatter_xs,
                metrics[algorithm].mses;
                color = colors[algorithm],
            )

            scatterlines!(
                axes.mse,
                line_xs,
                metrics[algorithm].mse_median,
                label = labels[algorithm],
                marker = 'x',
                color = colors[algorithm],
            )

            scatter_metric!(
                axes.nll,
                scatter_xs,
                metrics[algorithm].expected_nlls;
                color = colors[algorithm],
            )

            scatterlines!(
                axes.nll,
                line_xs,
                metrics[algorithm].expected_nll_median,
                label = labels[algorithm],
                marker = 'x',
                color = colors[algorithm],
            )
        end

        # fig[1, 1] = Legend(
        #     fig,
        #     axes.mse;
        #     nbanks = 3,
        #     framevisible = false,
        #     merge = true,
        #     unique = false,
        #     padding = (0.0f0, 0.0f0, 0.0f0, 0.0f0),
        #     tellheight = true,
        #     tellwidth = false,
        # )

        axislegend(axes.mse, position = legend_position)

        linkxaxes!(axes.mse, axes.nll)
        hidexdecorations!(axes.mse; ticks = false, grid = false)

        return (fig = fig, axes = axes)
    end
end

mse_lims = (low = 1.7, high = 4)
nll_lims = (low = 5e-1, high = 1e11)

begin
    plot = work_precision_plot(nll_scale = log10)

    ylims!(plot.axes.mse, mse_lims...)
    ylims!(plot.axes.nll; nll_lims...)

    safesave(plotsdir("ensemble_on_model", "work_precision_wall_time.pdf"), plot.fig)

    plot.fig
end

mse_lims_zoom = (low = 1.8, high = 1.9001)
nll_lims_zoom = (low = 1.61, high = 1.655)

begin
    plot = work_precision_plot(algorithms = [:cakf], legend_position = :rt)

    ylims!(plot.axes.mse, mse_lims_zoom...)
    ylims!(plot.axes.nll; nll_lims_zoom...)

    safesave(
        plotsdir("ensemble_on_model", "work_precision_wall_time_zoom_cakf.pdf"),
        plot.fig,
    )

    plot.fig
end

begin
    plot = work_precision_plot(
        abscissa = :rank,
        xlabel = "Rank",
        xscale = log2,
        nll_scale = log10,
    )

    ylims!(plot.axes.mse, mse_lims...)
    ylims!(plot.axes.nll; nll_lims...)

    safesave(plotsdir("ensemble_on_model", "work_precision_rank.pdf"), plot.fig)

    plot.fig
end

begin
    plot = work_precision_plot(
        algorithms = [:cakf],
        abscissa = :rank,
        xlabel = "Rank",
        xscale = log2,
        legend_position = :rt,
    )

    ylims!(plot.axes.mse, mse_lims_zoom...)
    ylims!(plot.axes.nll; nll_lims_zoom...)

    safesave(plotsdir("ensemble_on_model", "work_precision_rank_zoom_cakf.pdf"), plot.fig)

    plot.fig
end
