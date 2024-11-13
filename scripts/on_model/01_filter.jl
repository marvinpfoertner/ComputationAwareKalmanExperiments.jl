using DrWatson

@quickactivate "ComputationAwareKalmanExperiments"

include("common.jl")

rng = Random.seed!(seed + 1)

ys_train_aug = Vector{Union{eltype(ys_train),Missing}}(missing, length(ts))
ys_train_aug[ts_train_idcs] = ys_train

function cakf(dgmp, mmod, ys; rank, ts)
    fcache = ComputationAwareKalman.filter(
        dgmp,
        mmod,
        ys;
        update_kwargs = (max_iter = rank,),
        truncate_kwargs = (max_cols = rank,),
    )

    return [ComputationAwareKalman.interpolate(dgmp, fcache, t) for t in ts]
end
