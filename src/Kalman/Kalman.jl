module Kalman

using ComputationAwareKalman
using LinearAlgebra
using Statistics

import ..ComputationAwareKalmanExperiments: SquareRootGaussian

include("predict.jl")
include("update.jl")

include("filters.jl")

end

using .Kalman
