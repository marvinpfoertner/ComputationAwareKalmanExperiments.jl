using Kronecker

struct KroneckerGaussMarkovProcess{
    Tgmp<:ComputationAwareKalman.AbstractGaussMarkovProcess,
    Tmean_factor<:AbstractVector,
    Tcov_factor<:AbstractMatrix,
    Tlsqrt_cov_factor<:AbstractMatrix,
} <: ComputationAwareKalman.AbstractGaussMarkovProcess
    gmp::Tgmp

    mean_factor::Tmean_factor
    cov_factor::Tcov_factor
    lsqrt_cov_factor::Tlsqrt_cov_factor
end

function ComputationAwareKalman.statedim(gmp::KroneckerGaussMarkovProcess)
    return ComputationAwareKalman.statedim(gmp.gmp)
end

function ComputationAwareKalman.μ(gmp::KroneckerGaussMarkovProcess, t::Real)
    return kron(ComputationAwareKalman.μ(gmp.gmp, t), gmp.mean_factor)
end

function ComputationAwareKalman.Σ(gmp::KroneckerGaussMarkovProcess, t::Real)
    return kronecker(ComputationAwareKalman.Σ(gmp.gmp, t), gmp.cov_factor)
end

function ComputationAwareKalman.lsqrt_Σ(gmp::KroneckerGaussMarkovProcess, t::Real)
    return kronecker(ComputationAwareKalman.lsqrt_Σ(gmp.gmp, t), gmp.lsqrt_cov_factor)
end

function ComputationAwareKalman.A(gmp::KroneckerGaussMarkovProcess, t::Real, s::Real)
    return kronecker(ComputationAwareKalman.A(gmp.gmp, t, s), I(length(gmp.mean_factor)))
end

function transition_model(gmp::KroneckerGaussMarkovProcess, t::Real, s::Real)
    Ãₜₛ, b̃ₜₛ, Q̃ₜₛ = transition_model(gmp.gmp, t, s)

    return (
        kronecker(Ãₜₛ, I(length(gmp.mean_factor))),
        kron(b̃ₜₛ, gmp.mean_factor),
        kronecker(Q̃ₜₛ, gmp.cov_factor),
    )
end

function ComputationAwareKalman.A_b_lsqrt_Q(
    gmp::KroneckerGaussMarkovProcess,
    t::Real,
    s::Real,
)
    Ãₜₛ, b̃ₜₛ, lsqrt_Q̃ₜₛ = ComputationAwareKalman.A_b_lsqrt_Q(gmp.gmp, t, s)

    return (
        kronecker(Ãₜₛ, I(length(gmp.mean_factor))),
        kron(b̃ₜₛ, gmp.mean_factor),
        kronecker(lsqrt_Q̃ₜₛ, gmp.lsqrt_cov_factor),
    )
end
