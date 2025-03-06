using DrWatson
@quickactivate "ComputationAwareKalmanExperiments"

include("common.jl")

using GLMakie

function plot_states(model_and_data, states, t_idx = 1; gt = true, cred_int = true)
    @unpack H, ts, xs_factors, ts_train, xs_train, ys_train = model_and_data

    xs, = xs_factors

    states_plot = [H * state for state in states]

    means = hcat([Statistics.mean(state) for state in states_plot]...)'
    plot = surface(ts, xs, means, axis = (type = Axis3,))

    if cred_int
        stds = hcat([std(state) for state in states_plot]...)'
        surface!(ts, xs, means .+ 2 * stds, alpha = 0.4)
        surface!(ts, xs, means .- 2 * stds, alpha = 0.4)
    end

    scatter!(
        [
            Point3f(t, x, y) for (t, ys_train_t) in zip(ts_train, ys_train) for
            ((x,), y) in zip(xs_train, ys_train_t)
        ],
        color = :yellow,
    )

    if gt
        @unpack gt_fs = model_and_data

        gt_fs = hcat(gt_fs...)'
        surface!(ts, xs, gt_fs, colormap = :coolwarm)
    end

    return plot
end

seed = seeds[1]
algorithm = :kf_rts

model_and_data = build_model_and_data(seed)
res = run_algorithm(seed, algorithm, configs[algorithm][1], model_and_data = model_and_data);

@unpack smoother_states = res
plot_states(model_and_data, smoother_states, 21; gt = true, cred_int = false)
