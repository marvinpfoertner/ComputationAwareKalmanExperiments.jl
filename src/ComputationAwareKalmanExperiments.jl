module ComputationAwareKalmanExperiments

using ComputationAwareKalman
using CovarianceFunctions
using Kronecker
using LinearAlgebra
using Random
using Statistics

include("matfree/kernel_matrix.jl")
include("matfree/restriction.jl")

include("gmp/transition_model.jl")
include("gmp/kronecker.jl")
include("gmp/lti_sde.jl")
include("gmp/matern.jl")

include("cuda_utils.jl")
include("gaussians.jl")
include("metrics.jl")
include("sphere_utils.jl")

include("Kalman/Kalman.jl")
include("EnsembleKalman/EnsembleKalman.jl")

export MaternProcess

end
