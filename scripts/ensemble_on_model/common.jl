using BenchmarkTools
using ComputationAwareKalman
using ComputationAwareKalmanExperiments
import ComputationAwareKalmanExperiments: Kalman, EnsembleKalman, STSGMP_Matern
using Random
using Statistics

include("common/config.jl")
include("common/algorithms.jl")
include("common/experiment.jl")
