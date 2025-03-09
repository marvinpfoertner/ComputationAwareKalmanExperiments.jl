using DrWatson
@quickactivate "ComputationAwareKalmanExperiments"

include("common.jl")

run_all()

# Collect metrics
using DataFrames
using Statistics

df = collect_results(datadir("error_dynamics_on_model"), rinclude = [r"metrics_algorithm="])

function trajectory_median(trajectories)
    return Ref(median(hcat(trajectories...), dims = 2)[:, 1])
end

function collect_metrics(algorithm)
    df_algo = dropmissing(
        df[
            df.algorithm.==algorithm,
            [
                :seed,
                :rank,
                :mean_errors_filter,
                :cov_errors_filter,
                :mean_errors_smoother,
                :cov_errors_smoother,
            ],
        ],
    )
    df_metric_stats = combine(
        groupby(df_algo, :rank),
        :mean_errors_filter => Ref => :mean_errors_filter,
        :mean_errors_filter => trajectory_median => :mean_errors_filter_median,
        :cov_errors_filter => Ref => :cov_errors_filter,
        :cov_errors_filter => trajectory_median => :cov_errors_filter_median,
        :mean_errors_smoother => Ref => :mean_errors_smoother,
        :mean_errors_smoother => trajectory_median => :mean_errors_smoother_median,
        :cov_errors_smoother => Ref => :cov_errors_smoother,
        :cov_errors_smoother => trajectory_median => :cov_errors_smoother_median,
        :seed => Ref => :seeds,
    )

    return sort!(df_metric_stats, :rank)
end

metrics = collect_metrics(:cakf_caks)

# Generate plot
using CairoMakie
using TuePlots

CairoMakie.activate!(type = "svg")

function plot_error_dynamics(; smoother = true)
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

        supscript = smoother ? "s" : "{}"

        axes = (
            mean = Axis(
                fig[2, 1],
                xticksvisible = false,
                xticklabelsvisible = false,
                xgridvisible = false,
                xminorticks = ts_train_idcs,
                xminorgridvisible = true,
                ylabel = L"D^{-\frac{1}{2}} \cdot \Vert \mathbf{m}^%$supscript(t) - \hat{\mathbf{m}}^%$supscript(t) \Vert_2",
                yscale = log10,
            ),
            cov = Axis(
                fig[3, 1],
                xlabel = L"\text{Time}\ t",
                xticksvisible = false,
                xticklabelsvisible = false,
                xgridvisible = false,
                xminorticks = ts_train_idcs,
                xminorgridvisible = true,
                ylabel = L"D^{-1} \cdot \Vert \mathbf{P}^%$supscript(t) - \hat{\mathbf{P}}^%$supscript(t) \Vert_F",
                yscale = log10,
            ),
        )

        xs = 1:N_t

        colors = Makie.wong_colors()

        for (rank_idx, rank) in enumerate(metrics.rank)
            if smoother
                mean_errors = :mean_errors_smoother
                mean_errors_median = :mean_errors_smoother_median
                cov_errors = :cov_errors_smoother
                cov_errors_median = :cov_errors_smoother_median
            else
                mean_errors = :mean_errors_filter
                mean_errors_median = :mean_errors_filter_median
                cov_errors = :cov_errors_filter
                cov_errors_median = :cov_errors_filter_median
            end

            for (mean_error_traj, cov_error_traj) in
                zip(metrics[rank_idx, mean_errors], metrics[rank_idx, cov_errors])
                lines!(
                    axes.mean,
                    xs,
                    clamp.(sqrt.(mean_error_traj), 1e-30, Inf),
                    color = colors[rank_idx],
                    alpha = 0.4,
                )

                lines!(
                    axes.cov,
                    xs,
                    clamp.(cov_error_traj, 1e-30, Inf),
                    color = colors[rank_idx],
                    alpha = 0.4,
                )
            end

            lines!(
                axes.mean,
                xs,
                clamp.(sqrt.(metrics[rank_idx, mean_errors_median]), 1e-30, Inf),
                label = "$rank",
                color = colors[rank_idx],
            )

            lines!(
                axes.cov,
                xs,
                clamp.(metrics[rank_idx, cov_errors_median], 1e-30, Inf),
                label = "$rank",
                color = colors[rank_idx],
            )
        end

        fig[1, 1] = Legend(
            fig,
            axes.mean;
            nbanks = 4,
            framevisible = false,
            merge = true,
            unique = false,
            padding = (0.0f0, 0.0f0, 0.0f0, 0.0f0),
            tellheight = true,
        )

        linkxaxes!(axes.mean, axes.cov)

        ylims!(axes.mean; low = 1e-18, high = 1e1)
        ylims!(axes.cov; low = 1e-18, high = 1e1)

        return @ntuple(fig, axes)
    end
end

begin
    plot = plot_error_dynamics(smoother = true)

    safesave(plotsdir("error_dynamics_on_model", "error_dynamics_smoother.pdf"), plot.fig)

    plot.fig
end

begin
    plot = plot_error_dynamics(smoother = false)

    safesave(plotsdir("error_dynamics_on_model", "error_dynamics_filter.pdf"), plot.fig)

    plot.fig
end
