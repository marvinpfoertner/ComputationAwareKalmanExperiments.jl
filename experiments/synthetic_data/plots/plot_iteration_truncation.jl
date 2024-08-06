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
ys =
    hcat([[f_star(ts[i], xs[j]) for j = 1:length(xs)] for i = 1:length(ts)]...)' + rand(ϵ, (length(ts), length(xs)))


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
    fcache = ComputationAwareKalman.FilterCache()

    mₖ₋₁ = ComputationAwareKalman.μ(dgmp, 0)
    M⁺ₖ₋₁ = zeros(Float64, size(mₖ₋₁, 1), 0)

    for k = 1:length(dgmp)
        # Predict
        m⁻ₖ, M⁻ₖ = ComputationAwareKalman.predict(dgmp, k, mₖ₋₁, M⁺ₖ₋₁)

        # Update
        yₖ = ys[k, :]

        xₖ = ComputationAwareKalman.update(
            m⁻ₖ,
            ComputationAwareKalman.Σ(dgmp, k),
            M⁻ₖ,
            ComputationAwareKalman.H(mmod, k),
            ComputationAwareKalman.Λ(mmod, k),
            yₖ,
            abstol = 1e-9,
            reltol = 1e-9,
            max_iter = max_iter,
            # max_iter=size(ComputationAwareKalman.H(mmod, k), 1),
        )

        # Truncate
        M⁺ₖ, Π⁺ₖ = ComputationAwareKalman.truncate(xₖ.M, min_sval = 1e-6)

        push!(fcache; m⁻ = m⁻ₖ, xₖ..., M⁺ = M⁺ₖ, Π⁺ = Π⁺ₖ)

        mₖ₋₁ = xₖ.m
        M⁺ₖ₋₁ = M⁺ₖ
    end


    # Computation-aware Kalman Smoother
    scache =
        ComputationAwareKalman.smooth(dgmp, fcache, truncate_kwargs = (min_sval = 1e-9,))
    scache_truncated = ComputationAwareKalman.smooth(
        dgmp,
        fcache,
        truncate_kwargs = (max_cols = truncation_rank, min_sval = 1e-9),
        # truncate_kwargs=(min_sval=1e-9,),
    )

    return ts_plot, xs_plot, dgmp, H, fcache, scache, scache_truncated
end

# Plot
with_theme(T) do
    fig = Figure()

    max_iters_list = [1, 2, 3, 4]

    ax11 = Axis(fig[1, 1], ylabel = "Prediction")
    ax12 = Axis(fig[1, 2])
    ax13 = Axis(fig[1, 3])
    ax14 = Axis(fig[1, 4])
    ax21 = Axis(fig[2, 1], ylabel = "Uncertainty")
    ax22 = Axis(fig[2, 2])
    ax23 = Axis(fig[2, 3])
    ax24 = Axis(fig[2, 4])
    ax31 = Axis(fig[3, 1], xlabel = "Time", ylabel = "+ Truncation")
    ax32 = Axis(fig[3, 2], xlabel = "Time")
    ax33 = Axis(fig[3, 3], xlabel = "Time")
    ax34 = Axis(fig[3, 4], xlabel = "Time")

    hm_pred = nothing

    for (idx_col, max_iter) in enumerate(max_iters_list)

        # Computation-aware filtering and smoothing
        #TODO: same colormap limits for all plots
        #TODO: non-uniform mesh for more interesting uncertainty
        (ts_plot, xs_plot, dgmp, H_plot, fcache, scache, scache_truncated) =
            CAKS_prediction(max_iter = max_iter, truncation_rank = 2 * max_iter)


        # Posterior mean
        states =
            [ComputationAwareKalman.interpolate(dgmp, fcache, scache, t) for t in ts_plot]
        hm_pred = CairoMakie.contourf!(
            fig[1, idx_col],
            ts_plot,
            xs_plot,
            hcat([H_plot * Statistics.mean(states[k]) for k = 1:length(ts_plot)]...)',
            colormap = :coolwarm,
            levels = -0.5:0.1:1.1,
        )

        # Posterior standard deviation
        post_std =
            hcat(
                [
                    sqrt.(diag(H_plot * Statistics.cov(states[k]) * H_plot')) for
                    k = 1:length(ts_plot)
                ]...,
            )'
        states = [
            ComputationAwareKalman.interpolate(dgmp, fcache, scache_truncated, t) for
            t in ts_plot
        ]
        post_std_trunc =
            hcat(
                [
                    sqrt.(diag(H_plot * Statistics.cov(states[k]) * H_plot')) for
                    k = 1:length(ts_plot)
                ]...,
            )'
        print(maximum(hcat(post_std, post_std_trunc)))
        uq_scale_max = 0.76
        CairoMakie.contourf!(
            fig[2, idx_col],
            ts_plot,
            xs_plot,
            post_std,
            colormap = Colors.colormap("Purples"),
            levels = 0.0:0.1:uq_scale_max,
        )

        # Truncated posterior standard deviation
        CairoMakie.contourf!(
            fig[3, idx_col],
            ts_plot,
            xs_plot,
            post_std_trunc,
            colormap = Colors.colormap("Purples"),
            levels = 0.0:0.1:uq_scale_max,
        )

    end


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
        hidedecorations!(ax, label = false)
        tightlimits!(ax)
    end

    for row_idx in [1, 2, 3]
        Label(
            fig[row_idx, 5],
            "Space",
            font = Makie.to_font("Times"),
            tellwidth = true,
            tellheight = false,
            # padding=(0.0f0, 0.0f0, 0.0f0, 0.0f0),
            rotation = -π / 2,
            fontsize = 8,
        )
    end

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

    rowgap!(fig.layout, Fixed(3.0))
    colgap!(fig.layout, Fixed(3.0))

    save("$results_path/smoother_iter_trunc.pdf", fig)
    fig
end
