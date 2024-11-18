using Kronecker
using LinearAlgebra

# Dynamics model
stsgmp = ComputationAwareKalman.SpaceTimeSeparableGaussMarkovProcess(
    MaternProcess(matern_order_t, lₜ, σ²),
    _ -> zero(Float64),
    Σₓ,
)

# Discretization
ts = LinRange(temporal_domain..., Nₜ)
xs = LinRange(spatial_domain..., Nₓ)

spatial_cov_mat = ComputationAwareKalman.covariance_matrix(stsgmp.spatial_cov_fn, xs)

lsqrt_benchmark = @benchmarkable sqrt(spatial_cov_mat)
# tune!(lsqrt_benchmark)
lsqrt_benchmark_trial, lsqrt_spatial_cov_mat = BenchmarkTools.run_result(lsqrt_benchmark)
lsqrt_wall_time = median(lsqrt_benchmark_trial.times) / 1e9

gmp = ComputationAwareKalman.SpatiallyDiscretizedSTSGMP(
    stsgmp,
    xs,
    ComputationAwareKalman.mean_vector(stsgmp.spatial_mean_fn, xs),
    spatial_cov_mat,
    lsqrt_spatial_cov_mat,
)

H_all = kronecker(stsgmp.tgmp.H, I(Nₓ))

# Data
data, _ = produce_or_load(@dict(), datadir("on_model"), prefix = "data") do config
    rng = Random.seed!(data_seed)

    # Sample latent states
    gt_states = rand(rng, gmp, ts)

    # Sample measurements
    ys = [H_all * gt_state + sqrt(λ²) * randn(rng, Nₓ) for gt_state in gt_states]

    # Random subset as training data
    ts_train_idcs = sort!(randperm(rng, Nₜ)[1:Nₜ_train])
    xs_train_idcs = sort!(randperm(rng, Nₓ)[1:Nₓ_train])

    ys_train = [ys[i][xs_train_idcs] for i in ts_train_idcs]

    return @strdict gt_states ys ts_train_idcs xs_train_idcs ys_train
end

@unpack gt_states, ys, ts_train_idcs, xs_train_idcs, ys_train = data

fstars = [H_all * gt_state for gt_state in gt_states]

ts_train = ts[ts_train_idcs]
xs_train = xs[xs_train_idcs]

dgmp = ComputationAwareKalman.discretize(gmp, ts_train)

# Measurement model
H = kronecker(
    stsgmp.tgmp.H,
    ComputationAwareKalmanExperiments.RestrictionMatrix(Nₓ, xs_train_idcs),
)
Λ = λ² * I(Nₓ_train)

mmod = ComputationAwareKalman.UniformMeasurementModel(H, Λ)

# Discretize GMP at all `ts` and augment training data with missing values
# This is necessary for the ensemble Kalman methods to work
dgmp_aug = ComputationAwareKalman.discretize(gmp, ts)

ys_train_aug = Vector{Union{eltype(ys_train),Missing}}(missing, length(ts))
ys_train_aug[ts_train_idcs] = ys_train

