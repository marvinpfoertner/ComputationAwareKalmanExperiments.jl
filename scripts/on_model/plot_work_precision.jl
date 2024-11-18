using DrWatson
@quickactivate "ComputationAwareKalmanExperiments"

include("common.jl")  # TODO: Get rid of this import

run_all()

# Collect metrics
using DataFrames

df = collect_results(datadir("on_model", "results"), black_list = ["filter_states"])

function collect_deterministic_metrics(algorithm)
    df_algo =
        dropmissing(df[df.algorithm.==algorithm, [:rank, :mse, :expected_nll, :wall_time]])
    return sort!(df_algo, :rank)
end

function collect_stochastic_metrics(algorithm)
    df_algo = dropmissing(
        df[df.algorithm.==algorithm, [:rank, :seed, :mse, :expected_nll, :wall_time]],
    )
    df_metric_stats = combine(
        groupby(df_algo, :rank),
        :mse => Ref => :mses,
        :mse => median,
        :expected_nll => Ref => :expected_nlls,
        :expected_nll => median,
        :wall_time => median => :wall_time,
    )

    return sort!(df_metric_stats, :rank)
end

metrics = (
    kf = dropmissing(df[df.algorithm.=="kf", [:mse, :expected_nll, :wall_time]]),
    srkf = dropmissing(df[df.algorithm.=="srkf", [:mse, :expected_nll, :wall_time]]),
    enkf = collect_stochastic_metrics("enkf"),
    etkf = collect_deterministic_metrics("etkf"),
    cakf = collect_deterministic_metrics("cakf"),
)

# Generate plot
using CairoMakie
using TuePlots

CairoMakie.activate!(type = "svg")

labels = (kf = "KF", srkf = "SRKF", enkf = "EnKF", etkf = "ETKF", cakf = "CAKF")
colors = (kf = :violet, srkf = :green, enkf = :orange, etkf = :blue, cakf = :red)
isstochastic = (kf = false, srkf = false, enkf = true, etkf = false, cakf = false)

function scatter_stochastic_metric!(ax, xs, ys; markersize = 5, alpha = 0.5, kwargs...)
    scatter!(
        ax,
        vcat([repeat([x], length(y)) for (x, y) in zip(xs, ys)]...),
        vcat(ys...);
        markersize = markersize,
        alpha = alpha,
        kwargs...,
    )
end

function work_precision_plot(;
    abscissa = :wall_time,
    xlabel = "Wall Time [s]",
    xscale = log10,
    mse_scale = identity,
    nll_scale = identity,
    algorithms = [:enkf, :etkf, :cakf, :kf, :srkf],
)
    T = Theme(
        TuePlots.SETTINGS[:AISTATS];
        font = true,
        fontsize = true,
        single_column = true,
        figsize = true,
        thinned = true,
        nrows = 3,
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
                ylabel = "Expected NLL",
                yscale = nll_scale,
            ),
        )

        for algorithm in algorithms
            if isstochastic[algorithm]
                scatter_stochastic_metric!(
                    axes.mse,
                    metrics[algorithm][:, abscissa],
                    metrics[algorithm].mses;
                    color = colors[algorithm],
                )

                scatterlines!(
                    axes.mse,
                    metrics[algorithm][:, abscissa],
                    metrics[algorithm].mse_median,
                    label = labels[algorithm],
                    color = colors[algorithm],
                )

                scatter_stochastic_metric!(
                    axes.nll,
                    metrics[algorithm][:, abscissa],
                    metrics[algorithm].expected_nlls;
                    color = colors[algorithm],
                )

                scatterlines!(
                    axes.nll,
                    metrics[algorithm][:, abscissa],
                    metrics[algorithm].expected_nll_median,
                    label = labels[algorithm],
                    color = colors[algorithm],
                )
            elseif abscissa == :rank && (algorithm == :kf || algorithm == :srkf)
                hlines!(
                    axes.mse,
                    metrics[algorithm].mse,
                    label = labels[algorithm],
                    color = colors[algorithm],
                )
                hlines!(
                    axes.nll,
                    metrics[algorithm].expected_nll,
                    label = labels[algorithm],
                    color = colors[algorithm],
                )
            else
                scatterlines!(
                    axes.mse,
                    metrics[algorithm][:, abscissa],
                    metrics[algorithm].mse,
                    label = labels[algorithm],
                    color = colors[algorithm],
                )

                scatterlines!(
                    axes.nll,
                    metrics[algorithm][:, abscissa],
                    metrics[algorithm].expected_nll,
                    label = labels[algorithm],
                    color = colors[algorithm],
                )
            end
        end

        fig[3, 1] = Legend(
            fig,
            axes.mse;
            nbanks = 3,
            framevisible = false,
            merge = true,
            unique = false,
        )

        linkxaxes!(axes.mse, axes.nll)
        hidexdecorations!(axes.mse; ticks = false, grid = false)

        return (fig = fig, axes = axes)
    end
end

nll_lims = (low = -200.0, high = 3e3)

begin
    plot = work_precision_plot()

    ylims!(plot.axes.nll; nll_lims...)

    safesave(plotsdir("on_model", "work_precision_wall_time.pdf"), plot.fig)

    plot.fig
end

mse_lims_zoom = (low = 1.5e0, high = 1.72)
nll_lims_zoom = (low = 1.24, high = 1.57)

begin
    plot = work_precision_plot()

    ylims!(plot.axes.mse, mse_lims_zoom...)
    ylims!(plot.axes.nll; nll_lims_zoom...)

    safesave(plotsdir("on_model", "work_precision_wall_time_zoom_cakf.pdf"), plot.fig)

    plot.fig
end

begin
    plot = work_precision_plot(
        abscissa = :rank,
        xlabel = "Rank",
        xscale = log2,
        algorithms = [:enkf, :etkf, :cakf, :kf],
    )

    ylims!(plot.axes.nll; nll_lims...)

    safesave(plotsdir("on_model", "work_precision_rank.pdf"), plot.fig)

    plot.fig
end

begin
    plot = work_precision_plot(
        abscissa = :rank,
        xlabel = "Rank",
        xscale = log2,
        algorithms = [:enkf, :etkf, :cakf, :kf],
    )

    ylims!(plot.axes.nll; nll_lims_zoom...)

    safesave(plotsdir("on_model", "work_precision_rank_zoom_cakf.pdf"), plot.fig)

    plot.fig
end
