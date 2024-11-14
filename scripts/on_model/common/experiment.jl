function results(config::Dict)
    results, _ = produce_or_load(config, datadir("on_model", "results")) do config
        @unpack algorithm = config

        if "seed" in keys(config)
            rng = Random.seed!(config["seed"])
        end

        tic = Base.time_ns()

        if algorithm == "srkf"
            uᶠs = Kalman.srkf(dgmp_aug, mmod, ys_train_aug)
        elseif algorithm == "enkf"
            uᶠs = EnsembleKalman.enkf(
                dgmp_aug,
                mmod,
                ys_train_aug,
                rng = rng,
                rank = config["rank"],
            )
        elseif algorithm == "etkf"
            uᶠs = EnsembleKalman.etkf(
                dgmp_aug,
                mmod,
                ys_train_aug,
                rng = rng,
                rank = config["rank"],
            )
        elseif algorithm == "cakf"
            uᶠs = cakf(dgmp, mmod, ys_train; rank = config["rank"], ts = ts)
        end

        wall_time = (Base.time_ns() - tic) / 1e9

        mse = mean([
            ComputationAwareKalmanExperiments.mse(ustar, mean(uᶠ)) for
            (ustar, uᶠ) in zip(ustars, uᶠs)
        ])
        expected_nll = mean([
            mean(ComputationAwareKalmanExperiments.gaussian_nll.(ustar, mean(uᶠ), var(uᶠ))) for (ustar, uᶠ) in zip(ustars, uᶠs)
        ])

        return merge((@strdict uᶠs mse expected_nll wall_time), config)
    end

    return results
end