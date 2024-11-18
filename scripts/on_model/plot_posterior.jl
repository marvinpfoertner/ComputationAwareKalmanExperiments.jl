using DrWatson
@quickactivate "ComputationAwareKalmanExperiments"

include("common.jl")

using GLMakie

function plot_filter_states(filter_states; gt = true, cred_int = true)
    filter_states_plot = [H_all * filter_state for filter_state in filter_states]

    plot = scatter(ts_train, xs_train, hcat(ys_train...)')

    if gt
        surface!(ts, xs, hcat(fstars...)', colormap = :coolwarm)
    end

    means = hcat([mean(filter_state) for filter_state in filter_states_plot]...)'
    surface!(ts, xs, means)

    if cred_int
        stds = hcat([std(filter_state) for filter_state in filter_states_plot]...)'
        surface!(ts, xs, means .+ 2 * stds, alpha = 0.4)
        surface!(ts, xs, means .- 2 * stds, alpha = 0.4)
    end

    return plot
end

res = results(configs.etkf[5]);
@unpack filter_states = res
plot_filter_states(filter_states; gt = true, cred_int = true)
