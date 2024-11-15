module Kalman

using ComputationAwareKalman
using LinearAlgebra
using Statistics

using ..ComputationAwareKalmanExperiments
import ..ComputationAwareKalmanExperiments: Gaussian, SquareRootGaussian

include("predict.jl")
include("update.jl")

include("filters.jl")

include("interpolate.jl")

end

using .Kalman
