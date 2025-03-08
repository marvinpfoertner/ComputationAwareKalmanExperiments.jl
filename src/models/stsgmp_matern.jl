module STSGMP_Matern

using Adapt
using BenchmarkTools
using ComputationAwareKalman
using ComputationAwareKalmanExperiments
using CUDA
using DrWatson
using Kronecker
using LinearAlgebra
using Random

struct Matern32
    l::Float64
end

function (Σ::Matern32)(x1, x2)
    r = sqrt(sum((x1 .- x2) .^ 2)) / Σ.l
    return exp(-sqrt(3) * r) * (1 + sqrt(3) * r)
end

function build_dynamics_model(; parameters...)
    parameters = merge(
        (
            t_max = 5.0,
            d_x = 2,
            x_max = 20.0,
            nu_t = 1,
            l_t = 0.5,
            l_x = 0.5,
            sigma = 1.0,
            N_t = 100,
            N_x = 100,
        ),
        parameters,
    )

    @unpack t_max, d_x, x_max, nu_t, l_t, l_x, sigma, N_t, N_x = parameters

    stsgmp = ComputationAwareKalman.SpaceTimeSeparableGaussMarkovProcess(
        MaternProcess(nu_t, l_t, sigma),
        _ -> zero(Float64),
        Matern32(l_x),
    )

    # Discretize spatially
    xs_factors = [LinRange(0.0, x_max, N_x) for _ = 1:d_x]
    xs = collect(Iterators.product(xs_factors...))
    xs_flat = reshape(xs, :)

    spatial_cov_mat =
        ComputationAwareKalman.covariance_matrix(stsgmp.spatial_cov_fn, xs_flat)

    lsqrt_res, _ = produce_or_load(
        parameters,
        datadir("stsgmp_matern"),
        prefix = "lsqrt_spatial_cov_mat",
    ) do config
        lsqrt_benchmark = @benchmarkable begin
            spatial_cov_mat =
                CUDA.functional() ? CuArray(collect($spatial_cov_mat)) : $spatial_cov_mat
            spatial_cov_mat_eigen = eigen(Symmetric(spatial_cov_mat))
            eigenvals, eigenvecs = spatial_cov_mat_eigen
            lsqrt_spatial_cov_mat = eigenvecs * Diagonal(sqrt.(eigenvals))
            return lsqrt_spatial_cov_mat
        end
        # tune!(lsqrt_benchmark)
        lsqrt_benchmark_trial, lsqrt_spatial_cov_mat_dev =
            BenchmarkTools.run_result(lsqrt_benchmark)
        lsqrt_wall_time = median(lsqrt_benchmark_trial.times) / 1e9

        lsqrt_spatial_cov_mat =
            CUDA.functional() ? Array(lsqrt_spatial_cov_mat_dev) : lsqrt_spatial_cov_mat_dev

        return @strdict(lsqrt_spatial_cov_mat, lsqrt_wall_time)
    end

    @unpack lsqrt_spatial_cov_mat, lsqrt_wall_time = lsqrt_res

    gmp = ComputationAwareKalman.SpatiallyDiscretizedSTSGMP(
        stsgmp,
        xs_flat,
        ComputationAwareKalman.mean_vector(stsgmp.spatial_mean_fn, xs_flat),
        spatial_cov_mat,
        lsqrt_spatial_cov_mat,
    )

    H = kronecker(stsgmp.tgmp.H, I(length(xs_flat)))

    # Move spatially discretized model to GPU
    gmp_dev = ComputationAwareKalman.SpatiallyDiscretizedSTSGMP(
        stsgmp,
        xs_flat,
        ComputationAwareKalman.mean_vector(stsgmp.spatial_mean_fn, xs_flat),
        CUDA.functional() ? adapt(CuArray, spatial_cov_mat) : spatial_cov_mat,
        CUDA.functional() ?
        ComputationAwareKalmanExperiments.WrappedCuArray(
            adapt(CuArray, lsqrt_spatial_cov_mat),
        ) : lsqrt_spatial_cov_mat,
    )

    # Compile CUDA kernels
    gmp_dev.spatial_cov_mat * zeros(size(gmp_dev.spatial_cov_mat, 2))
    gmp_dev.spatial_cov_mat * zeros(size(gmp_dev.spatial_cov_mat, 2), 2)

    # Temporal discretization
    ts = LinRange(0.0, t_max, N_t)

    return @ntuple(stsgmp, xs_factors, xs, gmp, H, gmp_dev, ts, lsqrt_wall_time)
end

function model_and_data(
    data_seed;
    dynamics_model_parameters = (;),
    observation_model_parameters = (;),
)
    observation_model_parameters = merge(
        (ts_train_idcs = missing, N_t_train = 10, N_x_train = 400, lambda = 0.1),
        observation_model_parameters,
    )

    if !ismissing(observation_model_parameters.ts_train_idcs)
        observation_model_parameters = merge(
            observation_model_parameters,
            (N_t_train = length(observation_model_parameters.ts_train_idcs),),
        )
    end

    model = build_dynamics_model(; dynamics_model_parameters...)

    @unpack stsgmp, gmp, H, gmp_dev, ts = model

    data, _ = produce_or_load(
        merge(
            (; seed = data_seed),
            dynamics_model_parameters,
            observation_model_parameters,
        ),
        datadir("stsgmp_matern"),
        prefix = "data",
    ) do config
        @unpack ts_train_idcs, N_t_train, N_x_train, lambda = config

        data_seed = config.seed
        rng = Random.seed!(data_seed)

        # Sample latent states
        gt_states = rand(rng, gmp, ts)

        # Sample measurements
        ys = [
            H * gt_state + lambda * randn(rng, length(gmp.spatial_mean)) for
            gt_state in gt_states
        ]

        # Random subset as training data
        if ismissing(ts_train_idcs)
            ts_train_idcs = sort!(randperm(rng, length(gt_states))[1:N_t_train])
        end

        xs_train_idcs = sort!(randperm(rng, length(ys[1]))[1:N_x_train])

        return @strdict(data_seed, gt_states, ts_train_idcs, xs_train_idcs, ys)
    end

    @unpack lambda = observation_model_parameters
    @unpack gt_states, ts_train_idcs, xs_train_idcs, ys = data

    gt_fs = [H * gt_state for gt_state in gt_states]

    # Training set
    ts_train = ts[ts_train_idcs]
    xs_train = gmp.X[xs_train_idcs]
    ys_train = [ys[i][xs_train_idcs] for i in ts_train_idcs]

    # Discretize temporally on train set
    dgmp_train = ComputationAwareKalman.discretize(gmp, ts_train)
    dgmp_train_dev = ComputationAwareKalman.discretize(gmp_dev, ts_train)

    # Measurement model
    mmod = ComputationAwareKalman.UniformMeasurementModel(
        # H
        kronecker(
            stsgmp.tgmp.H,
            ComputationAwareKalmanExperiments.RestrictionMatrix(
                length(gmp.spatial_mean),
                xs_train_idcs,
            ),
        ),
        # Λ
        lambda^2 * I(length(xs_train)),
    )

    return merge(
        model,
        @ntuple(
            data_seed,
            gt_states,
            gt_fs,
            ts_train,
            xs_train,
            ys_train,
            dgmp_train,
            dgmp_train_dev,
            mmod
        )
    )
end

end

using .STSGMP_Matern
