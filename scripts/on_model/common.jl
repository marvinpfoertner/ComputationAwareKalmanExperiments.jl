using BenchmarkTools
using ComputationAwareKalman
using ComputationAwareKalmanExperiments
import ComputationAwareKalmanExperiments: Kalman, EnsembleKalman
using JLD2
using KernelFunctions
using Random
using Statistics

include("common/config.jl")
include("common/model_and_data.jl")
include("common/algorithms.jl")
include("common/experiment.jl")
