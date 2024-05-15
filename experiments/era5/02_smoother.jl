#!/usr/bin/env julia

config_id = length(ARGS) > 0 ? ARGS[1] : "debug"

include("init_experiment.jl")

fcache = jldopen(filter_res_path, "r") do results
    fcache = results["cache"]
    fcache.path = cache_path

    return fcache
end

scache = ComputationAwareKalman.JLD2SmootherCache{
    typeof(ComputationAwareKalman.m(fcache, length(dgmp))),
    typeof(ComputationAwareKalman.M(fcache, length(dgmp))),
    typeof(ComputationAwareKalman.w(fcache, length(dgmp))),
    typeof(ComputationAwareKalman.W(fcache, length(dgmp))),
}(cache_path, length(dgmp))

@withprogress name = "Smoothing..." begin
    tic = time_ns()

    ComputationAwareKalman.smooth!(
        dgmp,
        fcache,
        scache;
        truncate_kwargs=(
            max_cols=config.max_rank,
            min_sval=config.min_sval,
        ),
        callback_fn=((k, args...; kwargs...) -> @logprogress 1 - (k - 1) / (length(dgmp) - 1)),
    )

    global smoother_wall_ns = time_ns() - tic
end

jldsave(
    smoother_res_path;
    cache=scache,
    wall_time_ns=smoother_wall_ns,
)
