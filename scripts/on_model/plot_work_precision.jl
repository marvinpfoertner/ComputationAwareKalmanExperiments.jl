using DrWatson
@quickactivate "ComputationAwareKalmanExperiments"

include("common.jl")  # TODO: Get rid of this import

run_all()

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
        :mse => Ref => :mses,
        :mse => median,
        # :mse => (v -> quantile(v, 0.25)) => :mse_25,
        # :mse => (v -> quantile(v, 0.75)) => :mse_75,
        :expected_nll => Ref => :expected_nlls,
        :expected_nll => median,
        # :expected_nll => (v -> quantile(v, 0.25)) => :expected_nll_25,
        # :expected_nll => (v -> quantile(v, 0.75)) => :expected_nll_75,
        :wall_time => median => :wall_time,
    )

    return sort!(df_metric_stats, :rank)
end

metrics = Dict(
    "kf" => dropmissing(df[df.algorithm.=="kf", [:mse, :expected_nll, :wall_time]]),
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

# Plot parameters
rank_label = "Rank"
rank_xscale = log2

wall_time_label = "Wall Time [s]"
wall_time_xscale = log10

mse_label = "MSE"
mse_yscale = log10

nll_label = "Expected NLL"
nll_yscale = identity

alg2label = Dict(
    "kf" => "KF",
    "srkf" => "SRKF",
    "enkf" => "EnKF",
    "etkf" => "ETKF",
    "cakf" => "CAKF",
)
alg2color = Dict(
    "kf" => :violet,
    "srkf" => :green,
    "enkf" => :orange,
    "etkf" => :blue,
    "cakf" => :red,
)

function scatter_stochastic_metric!(
    ax,
    x_col::String,
    metric_col::String;
    algorithm::String = "enkf",
    markersize = 5,
    color = alg2color[algorithm],
    alpha = 0.5,
    scatter_kwargs...,
)
    scatter!(
        ax,
        vcat(
            [
                repeat([x], length(metrics[algorithm][i, metric_col])) for
                (i, x) in enumerate(metrics[algorithm][:, x_col])
            ]...,
        ),
        vcat(metrics[algorithm][:, metric_col]...),
        markersize = markersize,
        color = color,
        alpha = alpha,
        scatter_kwargs...,
    )
end

function work_precision_wall_time(;
    stochastic_algorithms = ["enkf"],
    deterministic_algorithms = ["etkf", "cakf", "kf", "srkf"],
)
    fig = Figure()

    ax_mse = Axis(
        fig[1, 1],
        xlabel = wall_time_label,
        xscale = wall_time_xscale,
        ylabel = mse_label,
        yscale = mse_yscale,
    )
    ax_nll = Axis(
        fig[2, 1],
        xlabel = wall_time_label,
        xscale = wall_time_xscale,
        ylabel = nll_label,
        yscale = nll_yscale,
    )

    # Stochastic algorithms
    for alg in stochastic_algorithms
        scatter_stochastic_metric!(ax_mse, "wall_time", "mses"; algorithm = alg)

        scatterlines!(
            ax_mse,
            metrics[alg].wall_time,
            metrics[alg].mse_median,
            label = alg2label[alg],
            color = alg2color[alg],
        )

        scatter_stochastic_metric!(ax_nll, "wall_time", "expected_nlls"; algorithm = alg)

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

    # ylims!(plot.ax_mse; low = 6e0, high = 5e2)
    # ylims!(plot.ax_nll; low = -300, high = 8e3)

    safesave(plotsdir("on_model", "work_precision_wall_time.pdf"), plot.fig)

    plot.fig
end

with_theme(T) do
    plot = work_precision_wall_time(stochastic_algorithms = [])

    ylims!(plot.ax_nll; low = 1.2, high = 2)

    safesave(plotsdir("on_model", "work_precision_wall_time_no_enkf.pdf"), plot.fig)

    plot.fig
end

function work_precision_rank(;
    kfs = ["kf"],
    stochastic_algorithms = ["enkf"],
    deterministic_algorithms = ["etkf", "cakf"],
)
    fig = Figure()

    ax_mse = Axis(
        fig[1, 1],
        xlabel = rank_label,
        xscale = rank_xscale,
        ylabel = mse_label,
        yscale = mse_yscale,
    )
    ax_nll = Axis(
        fig[2, 1],
        xlabel = rank_label,
        xscale = rank_xscale,
        ylabel = nll_label,
        yscale = nll_yscale,
    )

    for alg in kfs
        hlines!(ax_mse, metrics[alg].mse, label = alg2label[alg], color = alg2color[alg])
        hlines!(
            ax_nll,
            metrics[alg].expected_nll,
            label = alg2label[alg],
            color = alg2color[alg],
        )
    end

    # Stochastic algorithms
    for alg in stochastic_algorithms
        scatter_stochastic_metric!(ax_mse, "rank", "mses"; algorithm = alg)

        scatterlines!(
            ax_mse,
            metrics[alg].rank,
            metrics[alg].mse_median,
            label = alg2label[alg],
            color = alg2color[alg],
        )

        scatter_stochastic_metric!(ax_nll, "rank", "expected_nlls"; algorithm = alg)

        scatterlines!(
            ax_nll,
            metrics[alg].rank,
            metrics[alg].expected_nll_median,
            label = alg2label[alg],
            color = alg2color[alg],
        )
    end

    # Deterministic algorithms
    for alg in deterministic_algorithms
        scatterlines!(
            ax_mse,
            metrics[alg].rank,
            metrics[alg].mse,
            label = alg2label[alg],
            color = alg2color[alg],
        )

        scatterlines!(
            ax_nll,
            metrics[alg].rank,
            metrics[alg].expected_nll,
            label = alg2label[alg],
            color = alg2color[alg],
        )
    end

    axislegend(ax_mse)
    axislegend(ax_nll)

    linkxaxes!(ax_mse, ax_nll)
    hidexdecorations!(ax_mse; ticks = false, grid = false)

    return (fig = fig, ax_mse = ax_mse, ax_nll = ax_nll)
end

with_theme(T) do
    plot = work_precision_rank()

    # ylims!(plot.ax_mse; low = 5e0, high = 5e2)
    # ylims!(plot.ax_nll; low = -300, high = 8e3)

    safesave(plotsdir("on_model", "work_precision_rank.pdf"), plot.fig)

    plot.fig
end

with_theme(T) do
    plot = work_precision_rank(stochastic_algorithms = [])

    ylims!(plot.ax_nll; low = 1.2, high = 2)

    safesave(plotsdir("on_model", "work_precision_rank_no_enkf.pdf"), plot.fig)

    plot.fig
end
