abstract type AbstractStationaryLTISDESolution <:
              ComputationAwareKalman.AbstractGaussMarkovProcess end

function drift_matrix(::AbstractStationaryLTISDESolution) end
function stationary_covariance(::AbstractStationaryLTISDESolution) end

ComputationAwareKalman.statedim(gmp::AbstractStationaryLTISDESolution) =
    size(stationary_covariance(gmp), 2)

function ComputationAwareKalman.μ(gmp::AbstractStationaryLTISDESolution, t::AbstractFloat)
    F = drift_matrix(gmp)
    return zeros(promote_type(eltype(F), typeof(t)), size(F, 1))
end

function ComputationAwareKalman.Σ(gmp::AbstractStationaryLTISDESolution, ::AbstractFloat)
    return stationary_covariance(gmp)
end

function ComputationAwareKalman.A(
    gmp::AbstractStationaryLTISDESolution,
    t::AbstractFloat,
    s::AbstractFloat,
)
    F = drift_matrix(gmp)
    return exp(F * (t - s))
end

function transition_model(gmp::AbstractStationaryLTISDESolution, t::Real, s::Real)
    P∞ = stationary_covariance(gmp)

    Aₜₛ = ComputationAwareKalman.A(gmp, t, s)
    bₜₛ = zeros(eltype(Aₜₛ), size(Aₜₛ, 1))
    Qₜₛ = hermitianpart!(P∞ - Aₜₛ * P∞ * Aₜₛ')

    return Aₜₛ, bₜₛ, Qₜₛ
end

function ComputationAwareKalman.A_b_lsqrt_Q(
    gmp::AbstractStationaryLTISDESolution,
    t::AbstractFloat,
    s::AbstractFloat,
)
    Aₜₛ, bₜₛ, Qₜₛ = transition_model(gmp, t, s)

    return Aₜₛ, bₜₛ, sqrt(Qₜₛ)
end
