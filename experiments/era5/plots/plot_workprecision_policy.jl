#!/usr/bin/env julia

using CairoMakie
using TuePlots

CairoMakie.activate!(type = "svg")

T = Theme(
    TuePlots.SETTINGS[:ICML];
    font = true,
    fontsize = true,
    single_column = false,
    figsize = true,
    thinned = true,
    nrows = 3,
    ncols = 4,
)

include("../common.jl")

step_λθ = 12
budgets = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512]
policies = ["CG", "Random", "Coordinate"]
policy_to_suffix = Dict("CG" => "", "Random" => "-random", "Coordinate" => "-coordinate")

Λ = 0.1^2

with_theme(T) do
    fig = Figure()

    titlesize = 8
    xscale_time = log2
    yscale_mse = log10
    yscale_nll = identity
    xticks_time = LogTicks(0:8)
    ax_mse_kwargs = Dict(
        "Train" => (; yticks = LogTicks(-1:3),),
        "Test" => (; yticks = LogTicks(-1:3),),
    )
    ax_mse_ylims =
        Dict("Train" => (; low = 0.15, high = 750.0), "Test" => (; low = 1.1, high = 750.0))
    ax_nll_kwargs = Dict("Train" => (; yticks = 3:5,), "Test" => (; yticks = 3:5,))
    scatter_markersize = 8

    axs_nll_filter = Dict{String,Axis}()

    for (idx, dataset_train_or_test) in enumerate(["Train", "Test"])
        ax_mse_filter = Axis(
            fig[1, 2*idx-1];
            ylabel = dataset_train_or_test * " MSE",
            title = "CAKF",
            titlesize = titlesize,
            xscale = xscale_time,
            xticks = xticks_time,
            yscale = yscale_mse,
            ax_mse_kwargs[dataset_train_or_test]...,
        )
        ax_nll_filter = Axis(
            fig[2, 2*idx-1];
            xlabel = "Budget [iters/timestep]",
            ylabel = dataset_train_or_test * " NLL",
            xscale = xscale_time,
            xticks = xticks_time,
            yscale = yscale_nll,
            ax_nll_kwargs[dataset_train_or_test]...,
        )
        axs_nll_filter[dataset_train_or_test] = ax_nll_filter

        ax_mse_smoother = Axis(
            fig[1, 2*idx];
            title = "CAKS",
            titlesize = titlesize,
            xscale = xscale_time,
            xticks = xticks_time,
            yscale = yscale_mse,
            ax_mse_kwargs[dataset_train_or_test]...,
        )
        ax_nll_smoother = Axis(
            fig[2, 2*idx];
            xlabel = "Budget [iters/timestep]",
            xscale = xscale_time,
            xticks = xticks_time,
            yscale = yscale_nll,
            ax_nll_kwargs[dataset_train_or_test]...,
        )

        for policy in policies
            wall_times_filter = Float64[]
            wall_times_smoother = Float64[]
            mses_filter = Float64[]
            nlls_filter = Float64[]
            mses_smoother = Float64[]
            nlls_smoother = Float64[]

            for budget in budgets
                suffix = policy_to_suffix[policy]
                config = configs["$step_λθ-$budget$suffix"]

                jldopen("$(config.results_path)/metrics.jld2", "r") do results
                    push!(wall_times_filter, results["filter/wall_time_ns"] / 1e9 / 60)
                    push!(
                        mses_filter,
                        results["filter/mse_"*lowercase(dataset_train_or_test)],
                    )
                    push!(
                        nlls_filter,
                        results["filter/nll_"*lowercase(dataset_train_or_test)],
                    )
                    push!(wall_times_smoother, results["smoother/wall_time_ns"] / 1e9 / 60)
                    push!(
                        mses_smoother,
                        results["smoother/mse_"*lowercase(dataset_train_or_test)],
                    )
                    push!(
                        nlls_smoother,
                        results["smoother/nll_"*lowercase(dataset_train_or_test)],
                    )
                end
            end

            scatterlines!(
                ax_mse_filter,
                budgets,
                mses_filter,
                markersize = scatter_markersize,
                label = "$policy",
            )

            scatterlines!(
                ax_nll_filter,
                budgets,
                nlls_filter,
                markersize = scatter_markersize,
                label = "$policy",
            )

            scatterlines!(
                ax_mse_smoother,
                budgets,
                mses_smoother,
                markersize = scatter_markersize,
                label = "$policy",
            )
            scatterlines!(
                ax_nll_smoother,
                budgets,
                nlls_smoother,
                markersize = scatter_markersize,
                label = "$policy",
            )
        end

        ylims!(ax_mse_filter; ax_mse_ylims[dataset_train_or_test]...)

        linkxaxes!(ax_mse_filter, ax_nll_filter)
        linkxaxes!(ax_mse_smoother, ax_nll_smoother)
        linkyaxes!(ax_mse_filter, ax_mse_smoother)
        linkyaxes!(ax_nll_filter, ax_nll_smoother)
        hidexdecorations!(ax_mse_filter, grid = false)
        hidexdecorations!(ax_mse_smoother, grid = false)
        hideydecorations!(ax_mse_smoother, grid = false)
        hideydecorations!(ax_nll_smoother, grid = false)

        if idx == 2
            Legend(
                fig[3, :],
                ax_mse_filter,
                "Policy",
                framevisible = false,
                nbanks = 3,
                padding = (0.0f0, 0.0f0, 0.0f0, 0.0f0),
                tellheight = true,
                # titlegap=1.0,
                # rowgap=-2.0,
            )
        end


    end

    linkyaxes!(axs_nll_filter["Train"], axs_nll_filter["Test"])

    rowgap!(fig.layout, Fixed(10.0))
    colgap!(fig.layout, Fixed(10.0))

    # axislegend(ax_mse)
    # axislegend(ax_nll)

    save("$results_path/workprecision_policy.pdf", fig, update = false)
end
