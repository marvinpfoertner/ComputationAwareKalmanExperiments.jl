function results(config::Dict)
    results, _ = produce_or_load(config, datadir("on_model", "results")) do config
        @unpack algorithm = config

        if algorithm == "kf"
            filter_benchmark = @benchmarkable kf($dgmp, $mmod, $ys_train, $ts) evals = 1
        elseif algorithm == "srkf"
            filter_benchmark = @benchmarkable srkf($dgmp, $mmod, $ys_train, $ts) evals = 1
        elseif algorithm == "enkf"
            filter_benchmark = @benchmarkable enkf(
                $dgmp,
                $mmod,
                $ys_train,
                $ts;
                rng = Random.seed!($config["seed"]),
                rank = $(config["rank"]),
            ) evals = 1
        elseif algorithm == "etkf-truncate"
            filter_benchmark = @benchmarkable etkf_truncate(
                $dgmp,
                $mmod,
                $ys_train,
                $ts;
                rank = $(config["rank"]),
            ) evals = 1
        elseif algorithm == "etkf-lanczos"
            filter_benchmark = @benchmarkable etkf_lanczos(
                $dgmp,
                $mmod,
                $ys_train,
                $ts;
                rng = Random.seed!($config["seed"]),
                rank = $(config["rank"]),
            ) evals = 1
        elseif algorithm == "cakf"
            filter_benchmark = @benchmarkable cakf(
                $dgmp,
                $mmod,
                $ys_train,
                $ts;
                rank = $(config["rank"]),
            ) evals = 1
        end

        # tune!(filter_benchmark)
        benchmark_trial, filter_states = BenchmarkTools.run_result(filter_benchmark)

        mse = mean([
            ComputationAwareKalmanExperiments.mse(gt_state, mean(filter_state)) for
            (gt_state, filter_state) in zip(gt_states, filter_states)
        ])

        expected_nll = mean([
            mean(
                ComputationAwareKalmanExperiments.gaussian_nll.(
                    gt_state,
                    mean(filter_state),
                    var(filter_state),
                ),
            ) for (gt_state, filter_state) in zip(gt_states, filter_states)
        ])

        wall_time = median(benchmark_trial.times) / 1e9

        if algorithm == "srkf" || algorithm == "enkf"
            # SRKF and EnKF need access to a square root of the process noise covariance,
            # which is precomputed before the filters are run.
            wall_time += lsqrt_wall_time
        end

        return merge((@strdict filter_states mse expected_nll wall_time), config)
    end

    return results
end

function run_all()
    for algorithm in [:kf, :srkf, :enkf, :etkf_truncate, :etkf_lanczos, :cakf]
        for config in configs[algorithm]
            results(config)
        end
    end
end
