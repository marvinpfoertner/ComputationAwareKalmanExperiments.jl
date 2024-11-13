using KernelFunctions
using Kronecker
using LinearAlgebra

# Dynamics
lₜ = 0.5
lₓ = 1.5
σ² = 1.0

stsgmp = ComputationAwareKalman.SpaceTimeSeparableGaussMarkovProcess(
    MaternProcess(2, lₜ, σ²),
    _ -> zero(Float64),
    Matern52Kernel() ∘ ScaleTransform(1.0 / lₓ),
)

# Discretization
ts = LinRange(0.0, 5.0, 100)

Nₓ = 200
xs = LinRange(-1.0, 1.0, Nₓ)

dgmp = ComputationAwareKalman.discretize(stsgmp, ts, xs)

H_plot = kronecker(stsgmp.tgmp.H, I(Nₓ))

# Measurement
ts_train_idcs = 1:10:length(ts)
ts_train = ts[ts_train_idcs]

dgmp_train = ComputationAwareKalman.discretize(stsgmp, ts_train, xs)

xs_train_idcs = 1:10:Nₓ
Nₓ_train = length(xs_train_idcs)
xs_train = xs[xs_train_idcs]

λ² = 0.1^2

H = kronecker(
    stsgmp.tgmp.H,
    ComputationAwareKalmanExperiments.RestrictionMatrix(Nₓ, xs_train_idcs),
)
Λ = λ² * I(Nₓ_train)

mmod = ComputationAwareKalman.UniformMeasurementModel(H, Λ)
