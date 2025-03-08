function transition_model(gmc::ComputationAwareKalman.AbstractGaussMarkovChain, k::Integer)
    A, b, lsqrt_Q = ComputationAwareKalman.A_b_lsqrt_Q(gmc, k)
    return A, b, lsqrt_Q * lsqrt_Q'
end

function transition_model(
    gmp::ComputationAwareKalman.AbstractGaussMarkovProcess,
    t::Real,
    s::Real,
)
    A, b, lsqrt_Q = ComputationAwareKalman.A_b_lsqrt_Q(gmp, t, s)
    return A, b, lsqrt_Q * lsqrt_Q'
end

function transition_model(
    dgmp::ComputationAwareKalman.DiscretizedGaussMarkovProcess,
    k::Integer,
)
    return transition_model(dgmp.gmp, dgmp.ts[k+1], k > 0 ? dgmp.ts[k] : dgmp.ts[k+1])
end

function transition_model(gmp::ComputationAwareKalman.MaternProcess, t::Real, s::Real)
    Aₜₛ = ComputationAwareKalman.A(gmp, t, s)
    bₜₛ = zeros(eltype(Aₜₛ), size(Aₜₛ, 1))
    Qₜₛ = gmp.Σ∞ - Aₜₛ * gmp.Σ∞ * Aₜₛ'

    return Aₜₛ, bₜₛ, Qₜₛ
end

function transition_model(
    sdstsgmp::ComputationAwareKalman.SpatiallyDiscretizedSTSGMP,
    t::Real,
    s::Real,
)
    Ãₜₛ, b̃ₜₛ, Q̃ₜₛ = transition_model(sdstsgmp.stsgmp.tgmp, t, s)

    return (
        kronecker(Ãₜₛ, I(size(sdstsgmp.spatial_cov_mat, 1))),
        kron(b̃ₜₛ, sdstsgmp.spatial_mean),
        kronecker(Q̃ₜₛ, sdstsgmp.spatial_cov_mat),
    )
end
