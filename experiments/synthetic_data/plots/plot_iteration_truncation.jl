include("../common.jl")
using Makie
using CairoMakie
using TuePlots
using Colors

CairoMakie.activate!(type = "svg")

T = Theme(
    TuePlots.SETTINGS[:ICML];
    font = true,
    fontsize = true,
    single_column = true,
    figsize = true,
    thinned = true,
    nrows = 5,
    ncols = 4,
)


# Latent function
f_star(t, x) = sin(x) * exp(-t);

# Spatiotemporal grid
ts_plot = 0.0:0.02:1.0
xs_plot = 0.0:0.02:π

# Model
lₜ = 0.5
lₓ = 2.0
σ² = 1.0

# Σₜ = σ² * Matern32Kernel() ∘ ScaleTransform(1.0 / lₜ)
Σₓ = Matern52Kernel() ∘ ScaleTransform(1.0 / lₓ)
# Σ = tensor(Σₜ, Σₓ)

# f = GP(Σ)

# Data
ts = 0.0:0.1:1.0
xs = 0.0:0.2:π

xs_all = vcat(collect(xs), collect(xs_plot))

N = length(xs_all)
N_train = length(xs)
N_test = length(xs_plot)

# txs = ColVecs(
#     hcat(
#         [[ts[i], xs[j]] for i in 1:length(ts) for j in 1:length(xs)]...
#     )
# )

Random.seed!(123)

σ²obs = 0.1^2
ϵ = Normal(0.0, sqrt(σ²obs))
obs_noise = rand(ϵ, (length(ts), length(xs)))
ys = [[f_star(ts[i], xs[j]) + obs_noise[i, j] for j = 1:length(xs)] for i = 1:length(ts)]


function CAKS_prediction(; max_iter::Integer, truncation_rank::Integer)
    # Space-time separable Gauss-Markov process
    stsgmp = ComputationAwareKalman.SpaceTimeSeparableGaussMarkovProcess(
        MaternProcess(1, lₜ, σ²),
        _ -> zero(Float64),
        Σₓ,
    )

    dgmp = ComputationAwareKalman.discretize(stsgmp, collect(ts), xs_all)

    # Measurement Model
    mmod = ComputationAwareKalman.UniformMeasurementModel(
        kronecker(
            stsgmp.tgmp.H,
            ComputationAwareKalmanExperiments.RestrictionMatrix(N, 1:N_train),
        ),
        σ²obs * I(N_train),
    )

    # Emission Model
    H = kronecker(
        stsgmp.tgmp.H,
        ComputationAwareKalmanExperiments.RestrictionMatrix(N, N_train+1:N),
    )

    # Computation-aware Kalman Filter
    abstol = 1e-9
    reltol = 1e-9
    min_sval = 0.0

    fcache = ComputationAwareKalman.filter(
        dgmp,
        mmod,
        ys,
        update_kwargs = (abstol = abstol, reltol = reltol, max_iter = max_iter),
        truncate_kwargs = (max_cols = truncation_rank, min_sval = min_sval),
    )

    # Computation-aware Kalman Smoother
    scache = ComputationAwareKalman.smooth(
        dgmp,
        fcache,
        truncate_kwargs = (max_cols = truncation_rank, min_sval = min_sval),
    )

    return ts_plot, xs_plot, dgmp, H, fcache, scache
end

# Plot
with_theme(T) do
    fig = Figure()

    max_iters_list = [1, 2, 8, 16]

    ax11 = Axis(fig[1, 1], ylabel = "Prediction")
    ax12 = Axis(fig[1, 2])
    ax13 = Axis(fig[1, 3])
    ax14 = Axis(fig[1, 4], ylabel = "Space")
    ax21 = Axis(fig[2, 1], ylabel = "Uncertainty")
    ax22 = Axis(fig[2, 2])
    ax23 = Axis(fig[2, 3])
    ax24 = Axis(fig[2, 4], ylabel = "Space")
    ax31 = Axis(fig[3, 1], xlabel = "Time", ylabel = "Absolute Error")
    ax32 = Axis(fig[3, 2], xlabel = "Time")
    ax33 = Axis(fig[3, 3], xlabel = "Time")
    ax34 = Axis(fig[3, 4], xlabel = "Time", ylabel = "Space")

    for truncate in [false, true]

        post_mean_heatmap = nothing
        post_std_heatmap = nothing
        abs_err_heatmap = nothing

        for (idx_col, max_iter) in enumerate(max_iters_list)
            # Computation-aware filtering and smoothing
            #TODO: same colormap limits for all plots
            #TODO: non-uniform mesh for more interesting uncertainty
            (ts_plot, xs_plot, dgmp, H_plot, fcache, scache) = CAKS_prediction(
                max_iter = max_iter,
                truncation_rank = truncate ? 2 * max_iter : 10^6,
            )

            states = [
                ComputationAwareKalman.interpolate(dgmp, fcache, scache, t) for t in ts_plot
            ]

            # Posterior mean
            post_mean =
                hcat([H_plot * Statistics.mean(states[k]) for k = 1:length(ts_plot)]...)'

            post_mean_heatmap_tmp = CairoMakie.contourf!(
                fig[1, idx_col],
                ts_plot,
                xs_plot,
                post_mean,
                colormap = :coolwarm,
                levels = -0.5:0.05:1.1,
            )

            # Posterior standard deviation
            post_std =
                hcat(
                    [
                        sqrt.(diag(H_plot * Statistics.cov(states[k]) * H_plot')) for
                        k = 1:length(ts_plot)
                    ]...,
                )'

            uq_scale_max = 0.76

            post_std_heatmap_tmp = CairoMakie.contourf!(
                fig[2, idx_col],
                ts_plot,
                xs_plot,
                post_std,
                colormap = Colors.colormap("Purples"),
                levels = 0.0:0.05:uq_scale_max,
            )

            # Absolute error
            abs_err =
                abs.(
                    post_mean .-
                    hcat(
                        [
                            [f_star(ts_plot[i], xs_plot[j]) for j = 1:length(xs_plot)] for
                            i = 1:length(ts_plot)
                        ]...,
                    )'
                )

            abs_err_heatmap_tmp = CairoMakie.contourf!(
                fig[3, idx_col],
                ts_plot,
                xs_plot,
                abs_err,
                colormap = Colors.colormap("Purples"),
                levels = 0.0:0.05:uq_scale_max,
            )

            if idx_col == 1
                post_mean_heatmap = post_mean_heatmap_tmp
                post_std_heatmap = post_std_heatmap_tmp
                abs_err_heatmap = abs_err_heatmap_tmp
            end

            println(maximum(hcat(post_std, abs_err)))
        end

        # Color bars
        cbar_size = 6
        cbar_labelsize = 8
        cbar_ticklabelsize = 6
        cbar_ticksize = 3
        cbar_tickwidth = 0.5
        cbar_font = Makie.to_font("Times")
        Colorbar(
            fig[1, length(max_iters_list)+1],
            post_mean_heatmap,
            size = cbar_size,
            labelsize = cbar_labelsize,
            ticksize = cbar_ticksize,
            tickwidth = cbar_tickwidth,
            labelfont = cbar_font,
            ticklabelsize = cbar_ticklabelsize,
            ticklabelfont = cbar_font,
        )
        Colorbar(
            fig[2, length(max_iters_list)+1],
            post_std_heatmap,
            size = cbar_size,
            labelsize = cbar_labelsize,
            ticksize = cbar_ticksize,
            labelfont = cbar_font,
            ticklabelsize = cbar_ticklabelsize,
            ticklabelfont = cbar_font,
            tickwidth = cbar_tickwidth,
        )
        Colorbar(
            fig[3, length(max_iters_list)+1],
            abs_err_heatmap,
            size = cbar_size,
            labelsize = cbar_labelsize,
            ticksize = cbar_ticksize,
            labelfont = cbar_font,
            ticklabelsize = cbar_ticklabelsize,
            ticklabelfont = cbar_font,
            tickwidth = cbar_tickwidth,
        )


        # Link axes
        linkxaxes!(ax11, ax12)
        linkxaxes!(ax11, ax13)
        linkxaxes!(ax11, ax14)
        linkxaxes!(ax21, ax22)
        linkxaxes!(ax21, ax23)
        linkxaxes!(ax21, ax24)
        linkxaxes!(ax31, ax32)
        linkxaxes!(ax31, ax33)
        # linkxaxes!(ax31, ax24)
        # linkxaxes!(ax11, ax21)
        # linkxaxes!(ax11, ax31)


        for ax in [ax11, ax12, ax13, ax14, ax21, ax22, ax23, ax24, ax31, ax32, ax33, ax34]
            tightlimits!(ax)
        end

        for ax in [ax11, ax12, ax13, ax21, ax22, ax23]
            hidedecorations!(ax, label = false)
        end
        for ax in [ax31, ax32, ax33]
            hideydecorations!(ax, label = false)
        end
        for ax in [ax14, ax24]
            hidexdecorations!(ax, label = false)
        end
        for ax in [ax14, ax24, ax34]
            ax.yaxisposition = :right
        end
        # for row_idx in [1, 2, 3]
        #     Label(
        #         fig[row_idx, end-1],
        #         "Space",
        #         font = Makie.to_font("Times"),
        #         tellwidth = true,
        #         tellheight = false,
        #         # padding = (0.0f0, 0.0f0, 0.0f0, 0.0f0),
        #         rotation = -π / 2,
        #         fontsize = 8,
        #         halign = :left,
        #     )
        # end

        # timelabel = Label(
        #     fig[4, 2:3],
        #     "Time",
        #     font=Makie.to_font("Times"),
        #     tellwidth=true,
        #     padding=(0.0f0, 0.0f0, 0.0f0, 0.0f0),
        #     # rotation=-π / 2
        #     fontsize=8)

        iterationlabel = Label(
            fig[0, 2:3],
            "More Iterations →",
            font = Makie.to_font("Times"),
            tellwidth = true,
            padding = (0.0f0, 0.0f0, 0.0f0, 0.0f0),
            fontsize = 8,
        )

        rowgap!(fig.layout, Fixed(8.0))
        colgap!(fig.layout, Fixed(8.0))

        fname = truncate ? "trunc" : "full"
        save("$results_path/smoother_iter_$(fname).pdf", fig)
    end
end
