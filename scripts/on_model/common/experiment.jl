function run_filter(config::Dict)
    results, _ =
        produce_or_load(config, datadir("on_model", "results"), prefix = "states") do config
            @unpack algorithm = config

            if algorithm == "kf"
                filter_benchmark = @benchmarkable kf($dgmp, $mmod, $ys_train, $ts) evals = 1
            elseif algorithm == "srkf"
                filter_benchmark =
                    @benchmarkable srkf($dgmp, $mmod, $ys_train, $ts) evals = 1
            elseif algorithm == "enkf"
                filter_benchmark = @benchmarkable enkf(
                    $dgmp,
                    $dgmp_dev,
                    $mmod,
                    $ys_train,
                    $ts;
                    rng = Random.seed!($config["seed"]),
                    rank = $(config["rank"]),
                ) evals = 1
            elseif algorithm == "etkf-sample"
                filter_benchmark = @benchmarkable etkf_sample(
                    $dgmp,
                    $dgmp_dev,
                    $mmod,
                    $ys_train,
                    $ts;
                    rng = Random.seed!($config["seed"]),
                    rank = $(config["rank"]),
                ) evals = 1
            elseif algorithm == "etkf-lanczos"
                filter_benchmark = @benchmarkable etkf_lanczos(
                    $dgmp,
                    $dgmp_dev,
                    $mmod,
                    $ys_train,
                    $ts;
                    rng = Random.seed!($config["seed"]),
                    rank = $(config["rank"]),
                ) evals = 1
            elseif algorithm == "cakf"
                filter_benchmark = @benchmarkable cakf(
                    $dgmp,
                    $dgmp_dev,
                    $mmod,
                    $ys_train,
                    $ts;
                    rank = $(config["rank"]),
                ) evals = 1
            end

            # tune!(filter_benchmark)
            benchmark_trial, filter_states = BenchmarkTools.run_result(filter_benchmark)

            wall_time = median(benchmark_trial.times) / 1e9

            if algorithm == "srkf" || algorithm == "enkf" || algorithm == "etkf-sample"
                # SRKF, EnKF, and sample-based ETKF need access to a square root of the process noise covariance,
                # which is precomputed before the filters are run.
                wall_time += lsqrt_wall_time
            end

            return merge((@strdict filter_states wall_time), config)
        end

    return results
end

function compute_metrics(config::Dict)
    metrics, _ = produce_or_load(
        config,
        datadir("on_model", "results"),
        prefix = "metrics",
    ) do config
        filter_results = run_filter(config)

        @unpack filter_states, wall_time = filter_results

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

        return merge((@strdict wall_time mse expected_nll), config)
    end

    return metrics
end

function run_all(algorithms = [:enkf, :etkf_sample, :etkf_lanczos, :cakf])
    for algorithm in algorithms
        for config in configs[algorithm]
            compute_metrics(config)
        end
    end
end
