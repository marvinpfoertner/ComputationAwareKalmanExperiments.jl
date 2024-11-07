using WoodburyMatrices


function update_enkf(
    u⁻::EnsembleGaussian,
    y::AbstractVector,
    H::AbstractMatrix,
    Λ::AbstractVector,
    rng::Random.AbstractRNG,
)
    Eᶠ = members(u⁻)  # forecasting ensemble
    N = size(Eᶠ, 2)  # number of ensemble members
    Xᶠ = u⁻.Z  # forecasting ensemble-anomaly matrix

    Yₒ = y .+ sqrt(Λ) .* randn(rng, eltype(Λ), (length(y), N))  # perturbed observations

    Y = H * Xᶠ  # transformed ensemble-anomaly matrix
    C = SymWoodbury(Λ, Y, I)  # innovation matrix

    Eᵃ = Eᶠ + Xᶠ * Y' * (C \ (Yₒ - H * Eᶠ))  # analysis ensemble

    return EnsembleGaussian(Eᵃ)
end
