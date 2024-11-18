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
        elseif algorithm == "etkf"
            filter_benchmark = @benchmarkable etkf(
                $dgmp,
                $mmod,
                $ys_train,
                $ts;
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
        benchmark_trial, uᶠs = BenchmarkTools.run_result(filter_benchmark)

        mse = mean([
            ComputationAwareKalmanExperiments.mse(ustar, mean(uᶠ)) for
            (ustar, uᶠ) in zip(ustars, uᶠs)
        ])

        expected_nll = mean([
            mean(ComputationAwareKalmanExperiments.gaussian_nll.(ustar, mean(uᶠ), var(uᶠ))) for (ustar, uᶠ) in zip(ustars, uᶠs)
        ])

        wall_time = median(benchmark_trial.times) / 1e9

        if algorithm == "srkf" || algorithm == "enkf"
            # SRKF and EnKF need access to a square root of the process noise covariance,
            # which is precomputed before the filters are run.
            wall_time += lsqrt_wall_time
        end

        return merge((@strdict uᶠs mse expected_nll wall_time), config)
    end

    return results
end

function run_all()
    for algorithm in [:kf, :srkf]
        results(configs[algorithm])
    end

    for algorithm in [:enkf, :etkf, :cakf]
        for config in configs[algorithm]
            results(config)
        end
    end
end
