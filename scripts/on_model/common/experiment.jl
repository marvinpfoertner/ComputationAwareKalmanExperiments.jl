function run_experiment(config::Dict)
    results, _ = produce_or_load(config, datadir("on_model"), prefix = "results") do config
        @unpack algorithm = config

        if "seed" in keys(config)
            rng = Random.seed!(config["seed"])
        end

        if algorithm == "srkf"
            fstates = Kalman.srkf(dgmp, mmod, ys_train_aug)
        elseif algorithm == "enkf"
            fstates = EnsembleKalman.enkf(
                dgmp,
                mmod,
                ys_train_aug,
                rng = rng,
                rank = config["rank"],
            )
        elseif algorithm == "etkf"
            fstates = EnsembleKalman.etkf(
                dgmp,
                mmod,
                ys_train_aug,
                rng = rng,
                rank = config["rank"],
            )
        elseif algorithm == "cakf"
            fstates = cakf(dgmp_train, mmod, ys_train; rank = config["rank"], ts = ts)
        end

        mse = mean([
            ComputationAwareKalmanExperiments.mse(y, mean(fstate)) for
            (y, fstate) in zip(ys_test, fstates)
        ])
        expected_nll = mean([
            mean(
                ComputationAwareKalmanExperiments.gaussian_nll.(
                    y,
                    mean(fstate),
                    var(fstate),
                ),
            ) for (y, fstate) in zip(ys_test, fstates)
        ])

        return @strdict fstates mse expected_nll
    end

    @info config["algorithm"] results["mse"] results["expected_nll"]

    return results
end