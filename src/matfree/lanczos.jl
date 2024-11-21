using ElasticArrays

function lanczos(
    A,
    r₀::Tr₀;
    max_iter::Integer = length(r₀),
    tol = 1e-10,
) where {T<:AbstractFloat,Tr₀<:AbstractVector{T}}
    rᵢ₋₁ = r₀ / norm(r₀, 2)
    qᵢ₋₁ = zeros(T, length(r₀))

    rs =
        LinearAlgebra.QR(ElasticMatrix{T}(undef, length(r₀), 0), ElasticVector{T}(undef, 0))
    qs = ElasticMatrix{T}(undef, length(r₀), 0)
    αs = ElasticVector{T}(undef, 0)
    βs = ElasticVector{T}(undef, 0)

    for i = 1:max_iter
        # QR update
        w = deepcopy(rs.Q' * rᵢ₋₁)
        τ = LinearAlgebra.reflector!(view(w, i:length(w)))
        βᵢ₋₁ = w[i]

        if abs(βᵢ₋₁) <= tol
            break
        end

        append!(rs.factors, w)
        append!(rs.τ, τ)
        append!(βs, βᵢ₋₁)

        # Lanczos vector
        qᵢ = rs.Q[:, i]
        append!(qs, qᵢ)

        # Rayleigh quotient
        Aqᵢ = A(qᵢ)
        αᵢ = qᵢ' * Aqᵢ
        append!(αs, αᵢ)

        # Residual
        rᵢ = Aqᵢ - αᵢ * qᵢ - βᵢ₋₁ * qᵢ₋₁

        rᵢ₋₁ = rᵢ
        qᵢ₋₁ = qᵢ
    end

    return collect(qs), SymTridiagonal(collect(αs), collect(βs[2:end]))
end

function lanczos_lsqrt(args...; kwargs...)
    Q, T = lanczos(args...; kwargs...)

    return Q * sqrt(T)
end

