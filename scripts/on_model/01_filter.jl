using DrWatson

@quickactivate "ComputationAwareKalmanExperiments"

include("common.jl")

rng = Random.seed!(seed + 1)

ys_train_aug = Vector{Union{eltype(ys_train),Missing}}(missing, length(ts))
ys_train_aug[ts_train_idcs] = ys_train
