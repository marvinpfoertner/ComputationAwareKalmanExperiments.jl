module EnsembleKalman

using ComputationAwareKalman
using LinearAlgebra
using Random
using Statistics

import ..ComputationAwareKalmanExperiments: SquareRootGaussian

include("ensemble.jl")

include("initialize.jl")
include("predict.jl")
include("update.jl")

include("filters.jl")

end

using .EnsembleKalman
