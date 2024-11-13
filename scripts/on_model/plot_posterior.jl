using DrWatson

@quickactivate "ComputationAwareKalmanExperiments"

include("common.jl")
include("common/plots.jl")

res = results(configs["enkf"][2]);
fstates = res["fstates"]
plot_fstates(fstates; gt = true, cred_int = true)
