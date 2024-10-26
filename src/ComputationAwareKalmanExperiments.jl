module ComputationAwareKalmanExperiments

using ComputationAwareKalman
using CovarianceFunctions
using Kronecker
using LinearAlgebra
using Random

include("matfree/kernel_matrix.jl")
include("matfree/restriction.jl")

include("gmp/lti_sde.jl")
include("gmp/matern.jl")

include("sphere_utils.jl")

export MaternProcess

module EnsembleKalmanFilter

using ComputationAwareKalman
using Random

include("enkf/ensemble.jl")
include("enkf/predict.jl")

end

using .EnsembleKalmanFilter

end
