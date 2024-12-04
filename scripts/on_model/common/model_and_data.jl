using Adapt
using CUDA
using Kronecker
using LinearAlgebra

# Dynamics model
struct Matern32
    l::Float64
end

function (Σ::Matern32)(x1, x2)
    r = sqrt(sum((x1 .- x2) .^ 2)) / Σ.l
    return exp(-sqrt(3) * r) * (1 + sqrt(3) * r)
end

stsgmp = ComputationAwareKalman.SpaceTimeSeparableGaussMarkovProcess(
    MaternProcess(matern_order_t, lₜ, σ²),
    _ -> zero(Float64),
    Matern32(lₓ),
)

# Discretization
ts = LinRange(temporal_domain..., Nₜ)
xs_flat = [
    (x1, x2) for x1 in LinRange(spatial_domain[1]..., Nₓ[1]),
    x2 in LinRange(spatial_domain[2]..., Nₓ[2])
]
xs_flat = reshape(xs_flat, :)

spatial_cov_mat = ComputationAwareKalman.covariance_matrix(stsgmp.spatial_cov_fn, xs_flat)

lsqrt_benchmark = @benchmarkable begin
    spatial_cov_mat = CUDA.functional() ? CuArray(collect($spatial_cov_mat)) : $spatial_cov_mat
    spatial_cov_mat_eigen = eigen(Symmetric(spatial_cov_mat))
    eigenvals, eigenvecs = spatial_cov_mat_eigen
    lsqrt_spatial_cov_mat = eigenvecs * Diagonal(sqrt.(eigenvals))
    return CUDA.functional() ? ComputationAwareKalmanExperiments.WrappedCuArray(lsqrt_spatial_cov_mat) : lsqrt_spatial_cov_mat
end
# tune!(lsqrt_benchmark)
lsqrt_benchmark_trial, lsqrt_spatial_cov_mat_dev = BenchmarkTools.run_result(lsqrt_benchmark)
lsqrt_wall_time = median(lsqrt_benchmark_trial.times) / 1e9

lsqrt_spatial_cov_mat = CUDA.functional() ? Array(lsqrt_spatial_cov_mat_dev.cuA) : lsqrt_spatial_cov_mat_dev

gmp = ComputationAwareKalman.SpatiallyDiscretizedSTSGMP(
    stsgmp,
    xs_flat,
    ComputationAwareKalman.mean_vector(stsgmp.spatial_mean_fn, xs_flat),
    spatial_cov_mat,
    lsqrt_spatial_cov_mat,
)

H_all = kronecker(stsgmp.tgmp.H, I(length(xs_flat)))

# Move spatial covariance matrix to GPU
gmp_dev = ComputationAwareKalman.SpatiallyDiscretizedSTSGMP(
    stsgmp,
    xs_flat,
    ComputationAwareKalman.mean_vector(stsgmp.spatial_mean_fn, xs_flat),
    CUDA.functional() ? adapt(CuArray, spatial_cov_mat) : spatial_cov_mat,
    lsqrt_spatial_cov_mat_dev,
)
# Compile CUDA kernels
gmp_dev.spatial_cov_mat * zeros(size(gmp_dev.spatial_cov_mat, 2))
gmp_dev.spatial_cov_mat * zeros(size(gmp_dev.spatial_cov_mat, 2), 2)

# Data
data, _ = produce_or_load(@dict(), datadir("on_model"), prefix = "data") do config
    rng = Random.seed!(data_seed)

    # Sample latent states
    gt_states = rand(rng, gmp, ts)

    # Sample measurements
    ys = [
        H_all * gt_state + sqrt(λ²) * randn(rng, length(xs_flat)) for gt_state in gt_states
    ]

    # Random subset as training data
    ts_train_idcs = sort!(randperm(rng, Nₜ)[1:Nₜ_train])
    xs_train_idcs = sort!(randperm(rng, length(xs_flat))[1:Nₓ_train])

    ys_train = [ys[i][xs_train_idcs] for i in ts_train_idcs]

    return @strdict gt_states ys ts_train_idcs xs_train_idcs ys_train
end

@unpack gt_states, ys, ts_train_idcs, xs_train_idcs, ys_train = data

fstars = [H_all * gt_state for gt_state in gt_states]

ts_train = ts[ts_train_idcs]
xs_train = xs_flat[xs_train_idcs]

dgmp = ComputationAwareKalman.discretize(gmp, ts_train)
dgmp_dev = ComputationAwareKalman.discretize(gmp_dev, ts_train)

# Measurement model
H = kronecker(
    stsgmp.tgmp.H,
    ComputationAwareKalmanExperiments.RestrictionMatrix(length(xs_flat), xs_train_idcs),
)
Λ = λ² * I(Nₓ_train)

mmod = ComputationAwareKalman.UniformMeasurementModel(H, Λ)

# Discretize GMP at all `ts` and augment training data with missing values
# This is necessary for the ensemble Kalman methods to work
dgmp_aug = ComputationAwareKalman.discretize(gmp, ts)

ys_train_aug = Vector{Union{eltype(ys_train),Missing}}(missing, length(ts))
ys_train_aug[ts_train_idcs] = ys_train
