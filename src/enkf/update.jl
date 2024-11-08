using WoodburyMatrices


function update_enkf(
    u⁻::EnsembleGaussian,
    y::AbstractVector,
    H::AbstractMatrix,
    Λ::AbstractMatrix,
    rng::Random.AbstractRNG,
)
    # The implementation is based on section 4.1 the paper
    # Carrassi, A., Bocquet, M., Bertino, L., and Evensen, G. Data assimilation in the geosciences: An overview of methods, issues, and perspectives. WIREs Clim Change. 2018.

    Eᶠ = members(u⁻)  # forecasting ensemble, Eq. 32
    N = size(Eᶠ, 2)  # number of ensemble members, Eq. 32
    Xᶠ = u⁻.Z  # forecasting ensemble-anomaly matrix, Eq. 34

    Yₒ = y .+ sqrt(Λ) * randn(rng, eltype(Λ), (length(y), N))  # perturbed observations, Eqs. 37 & 38

    Y = H * Xᶠ  # transformed ensemble-anomaly matrix, Eq. 47
    C = SymWoodbury(Λ, Y, I(N))  # innovation matrix, Eq. 48 (and text below)

    Eᵃ = Eᶠ + Xᶠ * Y' * (C \ (Yₒ - H * Eᶠ))  # analysis ensemble, Eq. 44

    return EnsembleGaussian(Eᵃ)
end
