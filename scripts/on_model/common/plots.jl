using GLMakie

function plot_fstates(fstates; gt = true, cred_int = true)
    fstates_plot = [H_plot * fstate for fstate in fstates]

    plot = scatter(ts_train, xs_train, hcat(ys_train...)')

    if gt
        surface!(ts, xs, (H_plot * hcat(ys_test...))', colormap = :coolwarm)
    end

    means = hcat([mean(fstate) for fstate in fstates_plot]...)'
    surface!(ts, xs, means)

    if cred_int
        stds = hcat([std(fstate) for fstate in fstates_plot]...)'
        surface!(ts, xs, means .+ 2 * stds, alpha = 0.4)
        surface!(ts, xs, means .- 2 * stds, alpha = 0.4)
    end

    return plot
end
