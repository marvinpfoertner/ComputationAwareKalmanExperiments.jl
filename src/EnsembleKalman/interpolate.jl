struct EnsembleInterpolationGaussian{Tm<:AbstractVector,TZ<:AbstractMatrix,TQ}
    m::Tm
    Z::TZ
    Q::TQ
end

function EnsembleInterpolationGaussian(m; Q)
    return EnsembleInterpolationGaussian(m, Matrix{eltype(Q)}(undef, size(Q, 1), 0), Q)
end

function Base.:*(A::AbstractMatrix, u::EnsembleInterpolationGaussian)
    return EnsembleInterpolationGaussian(A * u.m, A * u.Z, A * u.Q * A')
end

Statistics.mean(u::EnsembleInterpolationGaussian) = u.m
Statistics.var(u::EnsembleInterpolationGaussian) = sum(u.Z .^ 2, dims = 2)[:, 1] + diag(u.Q)
Statistics.std(u::EnsembleInterpolationGaussian) = sqrt.(Statistics.var(u))

function interpolate(
    gmp::ComputationAwareKalman.AbstractGaussMarkovProcess,
    ts_data,
    filter_states_data::AbstractVector{<:SquareRootGaussian},
    t::Real,
)
    k = searchsortedlast(ts_data, t)

    if k < 1
        uᶠₜ = EnsembleInterpolationGaussian(
            ComputationAwareKalman.μ(gmp, t);
            Q = ComputationAwareKalman.Σ(gmp, t),
        )
    elseif t == ts_data[k]
        uᶠₜ = filter_states_data[k]
    else
        A, b, Q = ComputationAwareKalmanExperiments.transition_model(gmp, t, ts_data[k])

        uᶠₜ = EnsembleInterpolationGaussian(
            A * filter_states_data[k].m + b,
            A * filter_states_data[k].Z,
            Q,
        )
    end

    return uᶠₜ
end

function interpolate_lanczos(
    dgmp::ComputationAwareKalman.DiscretizedGaussMarkovProcess,
    uᶠs::AbstractVector{<:SquareRootGaussian},
    t::Real;
    rng::Random.AbstractRNG,
    rank::Integer,
)
    k = searchsortedlast(ComputationAwareKalman.ts(dgmp), t)

    if k < 1
        uᶠₜ = initialize_lanczos(dgmp.gmp, t; rng = rng, rank = rank)
    elseif t == ComputationAwareKalman.ts(dgmp)[k]
        uᶠₜ = uᶠs[k]
    else
        uᶠₜ = predict_lanczos(
            uᶠs[k],
            ComputationAwareKalmanExperiments.transition_model(
                dgmp.gmp,
                t,
                ComputationAwareKalman.ts(dgmp)[k],
            )...;
            rank = rank,
        )
    end

    return uᶠₜ
end
