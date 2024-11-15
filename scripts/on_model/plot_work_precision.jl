using DrWatson
@quickactivate "ComputationAwareKalmanExperiments"

include("common.jl")

# run_all()

# Collect metrics
using DataFrames

df = collect_results(datadir("on_model", "results"), black_list = ["uᶠs"])

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
        :mse => median,
        :mse => (v -> quantile(v, 0.25)) => :mse_25,
        :mse => (v -> quantile(v, 0.75)) => :mse_75,
        :expected_nll => median,
        :expected_nll => (v -> quantile(v, 0.25)) => :expected_nll_25,
        :expected_nll => (v -> quantile(v, 0.75)) => :expected_nll_75,
        :wall_time => median => :wall_time,
    )

    return sort!(df_metric_stats, :rank)
end

metrics = Dict(
    "srkf" => dropmissing(df[df.algorithm.=="srkf", [:mse, :expected_nll, :wall_time]]),
    "enkf" => collect_stochastic_metrics("enkf"),
    "etkf" => collect_deterministic_metrics("etkf"),
    "cakf" => collect_deterministic_metrics("cakf"),
)

# Generate plot
using CairoMakie
using TuePlots

CairoMakie.activate!(type = "svg")

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

alg2label = Dict("srkf" => "SRKF", "enkf" => "EnKF", "etkf" => "ETKF", "cakf" => "CAKF")
alg2color = Dict("srkf" => :black, "enkf" => :blue, "etkf" => :green, "cakf" => :red)
whisker_width = 4

function work_precision_wall_time(;
    stochastic_algorithms = ["enkf"],
    deterministic_algorithms = ["etkf", "cakf", "srkf"],
)
    fig = Figure()

    ax_mse = Axis(
        fig[1, 1],
        xlabel = "Wall Time [s]",
        ylabel = "MSE",
        xscale = log10,
        yscale = log10,
    )
    ax_nll =
        Axis(fig[2, 1], xlabel = "Wall Time [s]", ylabel = "Expected NLL", xscale = log10)

    # Stochastic algorithms
    for alg in stochastic_algorithms
        errorbars!(
            ax_mse,
            metrics[alg].wall_time,
            metrics[alg].mse_median,
            metrics[alg].mse_25,
            metrics[alg].mse_75;
            color = alg2color[alg],
            whiskerwidth = whisker_width,
        )

        scatterlines!(
            ax_mse,
            metrics[alg].wall_time,
            metrics[alg].mse_median,
            label = alg2label[alg],
            color = alg2color[alg],
        )

        errorbars!(
            ax_nll,
            metrics[alg].wall_time,
            metrics[alg].expected_nll_median,
            metrics[alg].expected_nll_25,
            metrics[alg].expected_nll_75;
            color = alg2color[alg],
            whiskerwidth = whisker_width,
        )

        scatterlines!(
            ax_nll,
            metrics[alg].wall_time,
            metrics[alg].expected_nll_median,
            label = alg2label[alg],
            color = alg2color[alg],
        )
    end

    # Deterministic algorithms
    for alg in deterministic_algorithms
        scatterlines!(
            ax_mse,
            metrics[alg].wall_time,
            metrics[alg].mse,
            label = alg2label[alg],
            color = alg2color[alg],
        )

        scatterlines!(
            ax_nll,
            metrics[alg].wall_time,
            metrics[alg].expected_nll,
            label = alg2label[alg],
            color = alg2color[alg],
        )
    end

    axislegend(ax_mse)
    axislegend(ax_nll)

    linkxaxes!(ax_mse, ax_nll)

    return (fig = fig, ax_mse = ax_mse, ax_nll = ax_nll)
end

with_theme(T) do
    plot = work_precision_wall_time()

    ylims!(plot.ax_mse; low = 5e0, high = 5e2)
    ylims!(plot.ax_nll; low = -300, high = 8e3)

    safesave(
        plotsdir("on_model", "work_precision_wall_time.pdf"),
        plot.fig,
        # update = false
    )

    plot.fig
end

with_theme(T) do
    plot = work_precision_wall_time(stochastic_algorithms = [])

    ylims!(plot.ax_nll; low = 1.68, high = 2)

    safesave(
        plotsdir("on_model", "work_precision_wall_time_no_enkf.pdf"),
        plot.fig,
        # update = false
    )

    plot.fig
end
