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

gmp = ComputationAwareKalman.discretize(stsgmp, xs)

H_all = kronecker(stsgmp.tgmp.H, I(Nₓ))

# Data
data, _ = produce_or_load(@dict(), datadir("on_model"), prefix = "data") do config
    rng = Random.seed!(data_seed)

    # Sample latent states
    ustars = rand(rng, gmp, ts)

    # Sample measurements
    ys = [H_all * ustar + sqrt(λ²) * randn(rng, Nₓ) for ustar in ustars]

    # Random subset as training data
    ts_train_idcs = sort!(randperm(rng, Nₜ)[1:Nₜ_train])
    xs_train_idcs = sort!(randperm(rng, Nₓ)[1:Nₓ_train])

    ys_train = [ys[i][xs_train_idcs] for i in ts_train_idcs]

    return @strdict ustars ys ts_train_idcs xs_train_idcs ys_train
end

@unpack ustars, ys, ts_train_idcs, xs_train_idcs, ys_train = data

fstars = [H_all * ustar for ustar in ustars]

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

