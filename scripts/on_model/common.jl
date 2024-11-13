using ComputationAwareKalman
using ComputationAwareKalmanExperiments
import ComputationAwareKalmanExperiments: Kalman, EnsembleKalman
using JLD2
using Random
using Statistics

include("common/config.jl")
include("common/model.jl")
include("common/data.jl")
include("common/algorithms.jl")
include("common/experiment.jl")
