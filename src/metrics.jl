function mse(y, m)
    return mean((y - m) .^ 2)
end

function gaussian_nll(y, μ, Σ)
    residual = y - μ
    return (residual' * (Σ \ residual) + logdet(2 * π * Σ)) / 2
end
