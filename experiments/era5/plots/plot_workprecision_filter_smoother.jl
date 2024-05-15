#!/usr/bin/env julia

using CairoMakie
using TuePlots

CairoMakie.activate!(type="svg")

T = Theme(
    TuePlots.SETTINGS[:ICML];
    font=true,
    fontsize=true,
    single_column=false,
    figsize=true,
    thinned=true,
    nrows=3,
    ncols=4,
)

include("../common.jl")

config_ids = Dict(
    24 => [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024],
    12 => [1, 2, 4, 8, 16, 32, 64, 128, 256, 512],
    6 => [1, 2, 4, 8, 16, 32, 64, 128, 256],
    # 3 => [64],
)

Λ = 0.1^2

with_theme(T) do
    fig = Figure()

    titlesize = 8
    xscale_time = log10
    yscale_mse = log10
    yscale_nll = identity
    xticks_time = exp10.(range(0, stop=3))
    yticks_train_mse = LogTicks(range(-4, stop=2))
    yticks_test_mse = LogTicks(range(0, stop=2))
    ylimits_test_mse = (10^(0 - 0.1), 10^(2 + 0.1))
    scatter_markersize = 8

    for (idx, dataset_train_or_test) in enumerate(["Train", "Test"])
        ax_mse_filter = Axis(
            fig[1, 2*idx-1],
            ylabel=dataset_train_or_test * " MSE",
            title="CAKF",
            titlesize=titlesize,
            xscale=xscale_time,
            yscale=yscale_mse,
            xticks=xticks_time,
            yticks=dataset_train_or_test == "Test" ? yticks_test_mse : yticks_train_mse,
        )
        ax_nll_filter = Axis(
            fig[2, 2*idx-1],
            xlabel="Wallclock Time [min]",
            ylabel=dataset_train_or_test * " NLL",
            xticks=xticks_time,
            xscale=xscale_time,
            yscale=yscale_nll,
        )

        ax_mse_smoother = Axis(
            fig[1, 2*idx],
            title="CAKS",
            titlesize=titlesize,
            xticks=xticks_time,
            xscale=xscale_time,
            yscale=yscale_mse,
            yticks=dataset_train_or_test == "Test" ? yticks_test_mse : yticks_train_mse,
        )
        ax_nll_smoother = Axis(
            fig[2, 2*idx],
            xlabel="Wallclock Time [min]",
            xticks=xticks_time,
            xscale=xscale_time,
            yscale=yscale_nll,
        )

        for step_λθ in sort(collect(keys(config_ids)), rev=true)
            wall_times_filter = Float64[]
            wall_times_smoother = Float64[]
            mses_filter = Float64[]
            nlls_filter = Float64[]
            mses_smoother = Float64[]
            nlls_smoother = Float64[]

            state_space_dim = nothing

            for budget in config_ids[step_λθ]
                config = configs["$step_λθ-$budget"]

                state_space_dim = jldopen("$(config.results_path)/metrics.jld2", "r") do results
                    push!(wall_times_filter, results["filter/wall_time_ns"] / 1e9 / 60)
                    push!(mses_filter, results["filter/mse_"*lowercase(dataset_train_or_test)])
                    push!(nlls_filter, results["filter/nll_"*lowercase(dataset_train_or_test)])
                    push!(wall_times_smoother, results["smoother/wall_time_ns"] / 1e9 / 60)
                    push!(mses_smoother, results["smoother/mse_"*lowercase(dataset_train_or_test)])
                    push!(nlls_smoother, results["smoother/nll_"*lowercase(dataset_train_or_test)])

                    return results["state_space_dim"]
                end
            end

            scatterlines!(
                ax_mse_filter,
                wall_times_filter,
                mses_filter,
                markersize=scatter_markersize,
                label="$state_space_dim",
            )

            scatterlines!(
                ax_nll_filter,
                wall_times_filter,
                nlls_filter,
                markersize=scatter_markersize,
                label="$state_space_dim",
            )

            scatterlines!(
                ax_mse_smoother,
                wall_times_smoother,
                mses_smoother,
                markersize=scatter_markersize,
                label="$state_space_dim",
            )
            scatterlines!(
                ax_nll_smoother,
                wall_times_smoother,
                nlls_smoother,
                markersize=scatter_markersize,
                label="$state_space_dim",
            )

            if dataset_train_or_test == "Train"
                # Noise level
                hlines!(
                    ax_mse_smoother,
                    Λ,
                    color=:gray,
                    linestyle=:dash,
                )
                text!(ax_mse_smoother, 10, 10^(-2.6), text="observation noise σ²", align=(:center, :center), color=:gray)
            end
        end

        linkxaxes!(ax_mse_filter, ax_nll_filter)
        linkxaxes!(ax_mse_smoother, ax_nll_smoother)
        linkyaxes!(ax_mse_filter, ax_mse_smoother)
        linkyaxes!(ax_nll_filter, ax_nll_smoother)
        hidexdecorations!(ax_mse_filter, grid=false)
        hidexdecorations!(ax_mse_smoother, grid=false)
        hideydecorations!(ax_mse_smoother, grid=false)
        hideydecorations!(ax_nll_smoother, grid=false)

        if dataset_train_or_test == "Test"
            ylims!(ax_mse_filter, ylimits_test_mse)
        end

        if idx == 2
            Legend(
                fig[3, :],
                ax_mse_filter,
                "State Space Dimension",
                framevisible=false,
                nbanks=3,
                padding=(0.0f0, 0.0f0, 0.0f0, 0.0f0),
                tellheight=true,
                # titlegap=1.0,
                # rowgap=-2.0,
            )
        end


    end

    rowgap!(fig.layout, Fixed(10.0))
    colgap!(fig.layout, Fixed(10.0))

    # axislegend(ax_mse)
    # axislegend(ax_nll)

    save("$results_path/workprecision_filter_smoother.pdf", fig, update=false)
end