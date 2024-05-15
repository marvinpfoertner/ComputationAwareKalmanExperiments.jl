using Adapt
using CUDA
using Kronecker
using LinearAlgebra

##########################################################################################
# Space-Time Separable Gauss-Markov Process Prior ########################################
##########################################################################################

struct ExtrinsicSphericalMatern32{Tl<:Real}
    l::Tl
end

function (Σ::ExtrinsicSphericalMatern32)(λ₁θ₁, λ₂θ₂)
    λ₁, θ₁ = λ₁θ₁
    λ₂, θ₂ = λ₂θ₂

    # GCS -> Cartesian
    x₁ = (
        r_earth * cos(λ₁) * cos(θ₁),
        r_earth * sin(λ₁) * cos(θ₁),
        r_earth * sin(θ₁),
    )
    x₂ = (
        r_earth * cos(λ₂) * cos(θ₂),
        r_earth * sin(λ₂) * cos(θ₂),
        r_earth * sin(θ₂),
    )

    # 3D Matérn-3/2 with lengthscale Σ.l
    r = sqrt(sum((x₁ .- x₂) .^ 2)) / Σ.l
    return exp(-sqrt(3) * r) * (1 + sqrt(3) * r)
end

function prior(lₜ::Float64, lₓ::Float64, σ²::Float64=1.0, pₜ::Integer=1)
    return ComputationAwareKalman.SpaceTimeSeparableGaussMarkovProcess(
        ComputationAwareKalmanExperiments.MaternProcess(pₜ, lₜ, σ²),
        _ -> 0.0,
        ExtrinsicSphericalMatern32(lₓ),
    )
end

function discretize_space(stsgmp::ComputationAwareKalman.SpaceTimeSeparableGaussMarkovProcess, λθs::Vector{Tuple{Float64,Float64}})
    spatial_mean_vec = stsgmp.spatial_mean_fn.(λθs)

    spatial_covmat = ComputationAwareKalmanExperiments.KernelMatrix{Float64}(stsgmp.spatial_cov_fn, λθs, λθs)

    if CUDA.functional()
        spatial_covmat = adapt(CuArray, spatial_covmat)
    end

    lsqrt_spatial_covmat = fill(NaN, size(spatial_covmat, 1), 1)  # sqrt(spatial_covmat)

    gmp = ComputationAwareKalman.SpatiallyDiscretizedSTSGMP(
        stsgmp,
        λθs,
        spatial_mean_vec,
        spatial_covmat,
        lsqrt_spatial_covmat,
    )

    H_plot = kronecker(stsgmp.tgmp.H, I(length(λθs)))

    return gmp, H_plot
end

##########################################################################################
# Measurement Model ######################################################################
##########################################################################################

function measurement_model(Hₜ::AbstractMatrix, era5_split::ERA5TrainTestSplit, σ²meas::Float64)
    H = kronecker(
        Hₜ,
        ComputationAwareKalmanExperiments.RestrictionMatrix(
            length(era5_split.era5.λθs),
            era5_split.λθ_idcs_train,
        )
    )

    Λ = σ²meas * I(size(H, 1))

    return ComputationAwareKalman.UniformMeasurementModel(H, Λ)
end
