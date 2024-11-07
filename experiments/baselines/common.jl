using ComputationAwareKalman
using ComputationAwareKalmanExperiments
using JLD2
using Random

include("common/model.jl")

data_path = joinpath(@__DIR__, "results", "data.jld2")


