seed = 2345

data, _ = produce_or_load(@dict(seed), datadir("on_model"), prefix = "data") do config
    @unpack seed = config

    rng = Random.seed!(seed)

    ys_test = rand(rng, dgmp, ts)

    ys_train = [
        rand(rng, mmod, k, y_test) for (k, y_test) in enumerate(ys_test[ts_train_idcs])
    ]

    return @strdict ys_test ys_train
end

@unpack ys_test, ys_train = data
