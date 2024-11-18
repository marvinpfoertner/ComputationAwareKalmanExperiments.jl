using DrWatson
@quickactivate "ComputationAwareKalmanExperiments"

include("common.jl")

using GLMakie

function plot_fstates(uᶠs; gt = true, cred_int = true)
    uᶠs_plot = [H_all * uᶠ for uᶠ in uᶠs]

    plot = scatter(ts_train, xs_train, hcat(ys_train...)')

    if gt
        surface!(ts, xs, hcat(fstars...)', colormap = :coolwarm)
    end

    means = hcat([mean(uᶠ) for uᶠ in uᶠs_plot]...)'
    surface!(ts, xs, means)

    if cred_int
        stds = hcat([std(uᶠ) for uᶠ in uᶠs_plot]...)'
        surface!(ts, xs, means .+ 2 * stds, alpha = 0.4)
        surface!(ts, xs, means .- 2 * stds, alpha = 0.4)
    end

    return plot
end

res = results(configs.etkf[5]);
@unpack uᶠs = res
plot_fstates(uᶠs; gt = true, cred_int = true)
