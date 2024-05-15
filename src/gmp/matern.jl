struct MaternProcess{TF,TP∞} <: AbstractStationaryLTISDESolution
    F::TF
    P∞::TP∞
    H::RestrictionMatrix{Vector{Int}}
end

function MaternProcess(k::Integer, l, σ²)
    λ = sqrt(2 * k + 1)

    F = [
        zeros(k) I(k)
        [-binomial(k + 1, i) * λ^(k + 1 - i) for i = 0:k]'
    ] ./ l

    if k == 0
        P∞ = σ² * I(1)
    elseif k == 1
        P∞ = σ² * Diagonal([1.0, λ^2])
    else
        L = [zeros(k); 1.0;;]
        Q = σ² * (2λ)^(2k + 1) / binomial(2k, k) / l * I(1)

        P∞ = Symmetric(lyap(F, L * Q * L'))
    end

    H = RestrictionMatrix(k + 1, [1])

    return MaternProcess{typeof(F),typeof(P∞)}(F, P∞, H)
end

function drift_matrix(gmp::MaternProcess)
    return gmp.F
end

function stationary_covariance(gmp::MaternProcess)
    return gmp.P∞
end
