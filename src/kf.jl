module KalmanFilter

function predict(
    m::AbstractVector,
    P::AbstractMatrix,
    A::AbstractMatrix,
    b::AbstractVector,
    Q::AbstractMatrix,
)
    m⁻ = A * m + b
    P⁻ = hermitianpart!(A * P * A' + Q)

    return m⁻, P⁻
end

function update(
    m⁻::AbstractVector,
    P⁻::AbstractMatrix,
    y::AbstractVector,
    H::AbstractMatrix,
    R::AbstractMatrix,
)
    v = y - H * m⁻
    S = hermitianpart!(H * P⁻ * H' + R)
    K = P⁻ * H' / S
    m = m⁻ + K * v
    P = hermitianpart!(P⁻ - K * H * P⁻)
    return m, P
end

end
