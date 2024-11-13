using DrWatson
@quickactivate "ComputationAwareKalmanExperiments"

include("common.jl")

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

# Assemble metrics
get_metrics(results) = (mse = results["mse"], expected_nll = results["expected_nll"])

metrics = Dict(
    "srkf" => get_metrics(results(configs["srkf"])),
    "enkf" => Dict(
        rank => get_metrics(results(config)) for (rank, config) in configs["enkf"]
    ),
    "etkf" => Dict(
        rank => get_metrics(results(config)) for (rank, config) in configs["etkf"]
    ),
    "cakf" => Dict(
        rank => get_metrics(results(config)) for (rank, config) in configs["cakf"]
    ),
)

with_theme(T) do
    fig = Figure()

    ax_mse = Axis(fig[1, 1], xlabel = "Rank", ylabel = "MSE", xscale = log2, yscale = log10)
    ax_nll = Axis(
        fig[2, 1],
        xlabel = "Rank",
        ylabel = "Expected NLL",
        xscale = log2,
        yscale = log10,
    )

    alg2label = Dict("srkf" => "SRKF", "enkf" => "EnKF", "etkf" => "ETKF", "cakf" => "CAKF")
    alg2color = Dict("srkf" => :black, "enkf" => :blue, "etkf" => :green, "cakf" => :red)

    hlines!(
        ax_mse,
        [metrics["srkf"].mse],
        label = alg2label["srkf"],
        color = alg2color["srkf"],
    )
    hlines!(
        ax_nll,
        [metrics["srkf"].expected_nll],
        label = alg2label["srkf"],
        color = alg2color["srkf"],
    )

    for alg in ["enkf", "etkf", "cakf"]
        scatterlines!(
            ax_mse,
            ranks,
            [metrics[alg][rank].mse for rank in ranks],
            label = alg2label[alg],
            color = alg2color[alg],
        )
        scatterlines!(
            ax_nll,
            ranks,
            [metrics[alg][rank].expected_nll for rank in ranks],
            label = alg2label[alg],
            color = alg2color[alg],
        )
    end

    axislegend(ax_mse)
    axislegend(ax_nll)

    safesave(
        plotsdir("on_model", "work_precision.pdf"),
        fig,
        # update = false
    )

    fig
end
