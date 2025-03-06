function build_model_and_data(seed::Integer)
    return STSGMP_Matern.model_and_data(
        seed + data_seed_offset;
        dynamics_model_parameters = (d_x = 1,),
        observation_model_parameters = (N_x_train = 40,),
    )
end

function run_algorithm(
    seed::Integer,
    algorithm::Symbol,
    algorithm_config::NamedTuple = (;);
    model_and_data = build_model_and_data(seed),
)
    @unpack dgmp_train, dgmp_train_dev, mmod, ys_train, ts = model_and_data

    results, _ = produce_or_load(
        merge(@ntuple(seed, algorithm), algorithm_config),
        datadir("error_dynamics_on_model"),
        prefix = "states",
    ) do config
        @unpack seed, algorithm = config

        if algorithm == :kf_rts
            states = kf_rts(dgmp_train, mmod, ys_train, ts)
        elseif algorithm == :cakf_caks
            states = cakf_caks(
                dgmp_train,
                dgmp_train_dev,
                mmod,
                ys_train,
                ts;
                rank = config.rank,
            )
        else
            error("Unknown algorithm: $algorithm")
        end

        return merge(tostringdict(ntuple2dict(states)), tostringdict(ntuple2dict(config)))
    end

    return dict2ntuple(results)
end

function compute_metrics(
    seed::Integer,
    algorithm::Symbol,
    algorithm_config::NamedTuple;
    model_and_data = build_model_and_data(seed),
)
    @unpack gt_states = model_and_data

    metrics, _ = produce_or_load(
        merge(@ntuple(seed, algorithm), algorithm_config),
        datadir("error_dynamics_on_model"),
        prefix = "metrics",
    ) do config
        kf_results = run_algorithm(seed, :kf_rts; model_and_data = model_and_data)
        algorithm_results = run_algorithm(
            seed,
            algorithm,
            algorithm_config;
            model_and_data = model_and_data,
        )

        rts_states = kf_results.smoother_states
        @unpack smoother_states = algorithm_results

        mses = [
            ComputationAwareKalmanExperiments.mse(mean(rts_state), mean(smoother_state)) for (rts_state, smoother_state) in zip(rts_states, smoother_states)
        ]

        # TODO: Frobenius norm between covariances

        return merge(@strdict(mses), tostringdict(ntuple2dict(config)))
    end

    return dict2ntuple(metrics)
end

function run_all(algorithms = [:kf_rts, :cakf_caks])
    for seed in seeds
        model_and_data = build_model_and_data(seed)

        for algorithm in algorithms
            for algorithm_config in configs[algorithm]
                compute_metrics(
                    seed,
                    algorithm,
                    algorithm_config;
                    model_and_data = model_and_data,
                )
            end
        end
    end
end
