function run_experiment(algorithm::String; rank::Integer)
    rng = Random.seed!(seed + 1)

    results, _ = produce_or_load(
        @dict(seed, algorithm),
        datadir("on_model"),
        prefix = "results",
    ) do config
        @unpack seed, algorithm = config

        if algorithm == "srkf"
            fstates = Kalman.srkf(dgmp, mmod, ys_train_aug)
        elseif algorithm == "enkf"
            fstates = EnsembleKalman.enkf(dgmp, mmod, ys_train_aug, rng = rng, rank = rank)
        elseif algorithm == "etkf"
            fstates = EnsembleKalman.etkf(dgmp, mmod, ys_train_aug, rng = rng, rank = rank)
        elseif algorithm == "cakf"
            fstates = cakf(dgmp_train, mmod, ys_train; rank = rank, ts = ts)
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

    @info algorithm results["mse"] results["expected_nll"]

    return results
end