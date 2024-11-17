module EnsembleKalman

using ComputationAwareKalman
using KrylovKit
using LinearAlgebra
using Random
using Statistics

using ..ComputationAwareKalmanExperiments
import ..ComputationAwareKalmanExperiments: SquareRootGaussian

include("ensemble.jl")

include("initialize.jl")
include("predict.jl")
include("update.jl")

include("filters.jl")

include("interpolate.jl")

include("truncate.jl")

end

using .EnsembleKalman
