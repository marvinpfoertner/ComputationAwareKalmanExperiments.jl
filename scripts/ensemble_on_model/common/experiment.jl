function build_model_and_data(seed::Integer)
    return STSGMP_2D_Matern.model_and_data(seed + data_seed_offset)
end

function run_filter(
    seed::Integer,
    algorithm::Symbol,
    algorithm_config::NamedTuple;
    model_and_data = build_model_and_data(seed),
)
    @unpack dgmp_train, dgmp_train_dev, mmod, ys_train, ts, lsqrt_wall_time = model_and_data

    results, _ = produce_or_load(
        merge(@ntuple(seed, algorithm), algorithm_config),
        datadir("ensemble_on_model"),
        prefix = "states",
    ) do config
        @unpack seed, algorithm = config

        if algorithm == :kf
            filter_benchmark =
                @benchmarkable kf($dgmp_train, $mmod, $ys_train, $ts) evals = 1
        elseif algorithm == :srkf
            filter_benchmark =
                @benchmarkable srkf($dgmp_train, $mmod, $ys_train, $ts) evals = 1
        elseif algorithm == :enkf
            filter_benchmark = @benchmarkable enkf(
                $dgmp_train,
                $dgmp_train_dev,
                $mmod,
                $ys_train,
                $ts;
                rng = $(Random.seed!(seed)),
                rank = $(config.rank),
            ) evals = 1
        elseif algorithm == :etkf_sample
            filter_benchmark = @benchmarkable etkf_sample(
                $dgmp_train,
                $dgmp_train_dev,
                $mmod,
                $ys_train,
                $ts;
                rng = $(Random.seed!(seed)),
                rank = $(config.rank),
            ) evals = 1
        elseif algorithm == :etkf_lanczos
            filter_benchmark = @benchmarkable etkf_lanczos(
                $dgmp_train,
                $dgmp_train_dev,
                $mmod,
                $ys_train,
                $ts;
                rng = $(Random.seed!(seed)),
                rank = $(config.rank),
            ) evals = 1
        elseif algorithm == :cakf
            filter_benchmark = @benchmarkable cakf(
                $dgmp_train,
                $dgmp_train_dev,
                $mmod,
                $ys_train,
                $ts;
                rank = $(config.rank),
            ) evals = 1
        end

        # tune!(filter_benchmark)
        benchmark_trial, filter_states = BenchmarkTools.run_result(filter_benchmark)

        wall_time = median(benchmark_trial.times) / 1e9

        if algorithm == :srkf || algorithm == :enkf || algorithm == :etkf_sample
            # SRKF, EnKF, and sample-based ETKF need access to a square root of the process noise covariance,
            # which is precomputed before the filters are run.
            wall_time += lsqrt_wall_time
        end

        return merge(@strdict(filter_states, wall_time), tostringdict(ntuple2dict(config)))
    end

    return results
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
        datadir("ensemble_on_model"),
        prefix = "metrics",
    ) do config
        filter_results = run_filter(
            seed,
            algorithm,
            algorithm_config;
            model_and_data = model_and_data,
        )

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

        return merge(
            (@strdict wall_time mse expected_nll),
            tostringdict(ntuple2dict(config)),
        )
    end

    return metrics
end

function run_all(algorithms = [:enkf, :etkf_sample, :etkf_lanczos, :cakf])
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
