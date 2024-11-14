function results(config::Dict)
    results, _ = produce_or_load(config, datadir("on_model", "results")) do config
        @unpack algorithm = config

        if "seed" in keys(config)
            rng = Random.seed!(config["seed"])
        end

        if algorithm == "srkf"
            filter_benchmark = @benchmarkable Kalman.srkf($dgmp_aug, $mmod, $ys_train_aug)
        elseif algorithm == "enkf"
            filter_benchmark = @benchmarkable EnsembleKalman.enkf(
                $dgmp_aug,
                $mmod,
                $ys_train_aug,
                rng = $rng,
                rank = $(config["rank"]),
            ) evals = 1
        elseif algorithm == "etkf"
            filter_benchmark = @benchmarkable EnsembleKalman.etkf(
                $dgmp_aug,
                $mmod,
                $ys_train_aug,
                rng = $rng,
                rank = $(config["rank"]),
            ) evals = 1
        elseif algorithm == "cakf"
            filter_benchmark = @benchmarkable cakf(
                $dgmp,
                $mmod,
                $ys_train;
                rank = $(config["rank"]),
                ts = $ts,
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

        if algorithm == "srkf" || algorithm == "enkf" || algorithm == "etkf"
            # SRKF, EnKF, and ETKF need access to a square root of the process noise covariance,
            # which is precomputed before the filters are run.
            wall_time += lsqrt_wall_time
        end

        return merge((@strdict uᶠs mse expected_nll wall_time), config)
    end

    return results
end

function run_all()
    results(configs["srkf"])

    for algorithm in ["enkf", "etkf"]
        for rank in ranks
            for config in configs[algorithm][rank]
                results(config)
            end
        end
    end

    for rank in ranks
        results(configs["cakf"][rank])
    end
end
