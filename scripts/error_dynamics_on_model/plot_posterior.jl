using DrWatson
@quickactivate "ComputationAwareKalmanExperiments"

include("common.jl")

using GLMakie

function plot_states(model_and_data, states, t_idx = 1; gt = true, cred_int = true)
    @unpack H, ts, xs, ts_train, xs_train, ys_train = model_and_data

    t = ts[t_idx]
    x1s = [x[1] for x in xs[:, 1]]
    x2s = [x[2] for x in xs[1, :]]
    filter_state_plot = H * states[t_idx]

    mean = reshape(Statistics.mean(filter_state_plot), (length(x1s), length(x2s)))
    plot = surface(x1s, x2s, mean, axis = (type = Axis3,))

    if cred_int
        std = reshape(Statistics.std(filter_state_plot), (length(x1s), length(x2s)))
        surface!(x1s, x2s, mean .+ 2 * std, alpha = 0.4)
        surface!(x1s, x2s, mean .- 2 * std, alpha = 0.4)
    end

    k = searchsortedfirst(ts_train, t)
    if k <= lastindex(ts_train)
        scatter!(
            [Point3f(x1, x2, y) for ((x1, x2), y) in zip(xs_train, ys_train[k])],
            color = :yellow,
        )
    end

    if gt
        @unpack gt_fs = model_and_data

        gt_f = reshape(gt_fs[t_idx], (length(x1s), length(x2s)))
        surface!(x1s, x2s, gt_f, colormap = :coolwarm)
    end

    return plot
end

seed = seeds[1]
algorithm = :kf_rts

model_and_data = build_model_and_data(seed)
res = run_algorithm(seed, algorithm, configs[algorithm][1], model_and_data = model_and_data);

@unpack smoother = res
plot_filter_states(model_and_data, filter_states, 21; gt = true, cred_int = false)
