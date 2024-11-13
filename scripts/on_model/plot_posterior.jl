using DrWatson
@quickactivate "ComputationAwareKalmanExperiments"

include("common.jl")
include("common/plots.jl")

res = results(configs["enkf"][10]);
@unpack uᶠs = res
plot_fstates(uᶠs; gt = true, cred_int = false)
