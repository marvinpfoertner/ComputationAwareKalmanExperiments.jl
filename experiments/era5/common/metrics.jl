function gaussian_nll(y, μ, σ²)
    return 0.5 * ((y - μ)^2 / σ² + log(2.0 * π * σ²))
end
