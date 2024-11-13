using WoodburyMatrices


function update_enkf(
    u⁻::SquareRootGaussian,
    y::AbstractVector,
    H::AbstractMatrix,
    Λ::AbstractMatrix,
    rng::Random.AbstractRNG,
)
    # This implementation is based on section 4.1 of the paper
    # Carrassi, A., Bocquet, M., Bertino, L., and Evensen, G. "Data assimilation in the geosciences: An overview of methods, issues, and perspectives." WIREs Clim Change. 2018.

    Eᶠ = ensemble(u⁻)  # forecasting ensemble, Eq. 32
    N = size(Eᶠ, 2)  # number of ensemble members, Eq. 32
    Xᶠ = u⁻.Z  # forecasting ensemble-anomaly matrix, Eq. 34

    Yₒ = y .+ sqrt(Λ) * randn(rng, eltype(Λ), (length(y), N))  # perturbed observations, Eqs. 37 & 38

    Y = H * Xᶠ  # transformed ensemble-anomaly matrix, Eq. 47
    C = SymWoodbury(Λ, Y, I(N))  # innovation matrix, Eq. 48 (and text below)

    Eᵃ = Eᶠ + Xᶠ * (Y' * (C \ (Yₒ - H * Eᶠ)))  # analysis ensemble, Eq. 44

    return ensemble_to_gaussian(Eᵃ)
end


function update_etkf(
    u⁻::SquareRootGaussian,
    y::AbstractVector,
    H::AbstractMatrix,
    Λ::AbstractMatrix,
)
    # This implementation is based on section 3.a of the paper
    # Tippett, M. K., Anderson, J. L., Bishop, C. H., Hamill, T. M., Whitaker, J. S. "Ensemble square root filters." Mon. Wea. Rev., 131, 1485-1490, 2003.

    Zᶠ = u⁻.Z  # square root of forecast covariance
    HZᶠ = H * Zᶠ  # square root of transformed forecast covariance

    γs, C = eigen!(hermitianpart!(HZᶠ' * (Λ \ HZᶠ)))
    Zᵃ = Zᶠ * C * Diagonal(1 ./ sqrt.(γs .+ 1.0))  # square root of analysis covariance, Eq. 16

    return SquareRootGaussian(__update_mean(u⁻, y, H, Λ, HZᶠ), Zᵃ)
end

function __update_mean(
    u⁻::SquareRootGaussian,
    y::AbstractVector,
    H::AbstractMatrix,
    Λ::AbstractMatrix,
    HZ::AbstractMatrix = H * u⁻.Z,
)
    S = SymWoodbury(Λ, HZ, I(size(u⁻.Z, 2)))  # innovation covariance

    return u⁻.m + u⁻.Z * (HZ' * (S \ (y - H * u⁻.m)))
end
