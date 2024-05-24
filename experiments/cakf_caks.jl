### A Pluto.jl notebook ###
# v0.19.40

using Markdown
using InteractiveUtils

# ╔═╡ 3438fb37-dab7-4bea-bf97-e1b087918e98
begin
    import Pkg
    Pkg.activate("../")
    Pkg.instantiate()
end

# ╔═╡ fb31863e-9027-11ed-066a-7fc870cb91db
begin
    using AbstractGPs
    using ComputationAwareKalman
    using ComputationAwareKalmanExperiments
    using GaussianDistributions
    using KernelFunctions
    using Kronecker
    using LinearAlgebra
    using Plots
    using Random
    using Statistics
end

# ╔═╡ e79effe2-213b-49ed-9ce9-0b4d48a2f190
f_star(t, x) = sin(x) * exp(-t);

# ╔═╡ bbf87054-1e15-4fcb-a65f-7ec42018eeb6
begin
    ts_plot = 0.0:0.01:1.0
    xs_plot = 0.0:0.05:π

    @info size(xs_plot)

    plot(ts_plot, xs_plot, f_star, st = [:surface])
end

# ╔═╡ ca0bc9de-32e8-4fcc-acd6-46c5b522c99e
begin
    lₜ = 0.5
    lₓ = 2.0
    σ² = 1.0

    Σₜ = σ² * Matern32Kernel() ∘ ScaleTransform(1.0 / lₜ)
    Σₓ = Matern52Kernel() ∘ ScaleTransform(1.0 / lₓ)
    Σ = tensor(Σₜ, Σₓ)
end

# ╔═╡ 4633f1f9-97ac-4e03-ac1e-3a87f90fb879
f = GP(Σ);

# ╔═╡ 3f0793a0-e993-4e1d-bf77-5c2940832ab3
begin
    ts = 0.0:0.05:1.0
    xs = 0.0:0.1:π

    @info size(ts, 1), size(xs, 1)

    txs = ColVecs(hcat([[ts[i], xs[j]] for i = 1:length(ts) for j = 1:length(xs)]...))
    ys = hcat([[f_star(ts[i], xs[j]) for j = 1:length(xs)] for i = 1:length(ts)]...)'

    nothing
end

# ╔═╡ cba9c673-aa11-4226-ad73-d36133f78fde
begin
    σ²obs = 0.3^2

    ftxs = f(txs, σ²obs)

    nothing
end

# ╔═╡ 93346804-c558-49d5-9872-c11d9ed6e258
p_ftxs = posterior(ftxs, reshape(ys', length(ys)));

# ╔═╡ abe83b9d-f73f-4d54-b7ac-f8d953dfc5ad
begin
    txs_plot = ColVecs(
        hcat(
            [
                [ts_plot[i], xs_plot[j]] for i = 1:length(ts_plot) for j = 1:length(xs_plot)
            ]...,
        ),
    )

    p_marginals = marginals(p_ftxs(txs_plot))

    plot(
        ts_plot,
        xs_plot,
        reshape(mean.(p_marginals), length(xs_plot), length(ts_plot)),
        st = :surface,
        colorbar = false,
    )

    plot!(
        ts_plot,
        xs_plot,
        reshape(quantile.(p_marginals, 0.95), length(xs_plot), length(ts_plot)),
        st = :surface,
        fa = 0.7,
    )
end

# ╔═╡ e6cbe302-a906-4dcf-9084-868c78343dcf
begin
    xs_all = vcat(collect(xs), collect(xs_plot))

    N = length(xs_all)
    N_train = length(xs)
    N_test = length(xs_plot)

    nothing
end

# ╔═╡ 3e129fc5-a5b9-4d74-9470-5c83ee831c59
begin
    stsgmp = ComputationAwareKalman.SpaceTimeSeparableGaussMarkovProcess(
        MaternProcess(1, lₜ, σ²),
        _ -> zero(Float64),
        Σₓ,
    )

    dgmp = ComputationAwareKalman.discretize(stsgmp, collect(ts), xs_all)

    # Measurement Model
    mmod = ComputationAwareKalman.UniformMeasurementModel(
        kronecker(
            stsgmp.tgmp.H,
            ComputationAwareKalmanExperiments.RestrictionMatrix(N, 1:N_train),
        ),
        σ²obs * I(N_train),
    )

    # Emission Model
    H_plot = kronecker(
        stsgmp.tgmp.H,
        ComputationAwareKalmanExperiments.RestrictionMatrix(N, N_train+1:N),
    )

    nothing
end

# ╔═╡ 7d149836-b07b-468a-bdb0-8d8cae45ae23
begin
    fcache = ComputationAwareKalman.FilterCache()

    mₖ₋₁ = ComputationAwareKalman.μ(dgmp, 0)
    M⁺ₖ₋₁ = zeros(Float64, size(mₖ₋₁, 1), 0)

    for k = 1:length(dgmp)
        # Predict
        m⁻ₖ, M⁻ₖ = ComputationAwareKalman.predict(dgmp, k, mₖ₋₁, M⁺ₖ₋₁)

        # Update
        yₖ = ys[k, :]

        xₖ = ComputationAwareKalman.update(
            m⁻ₖ,
            ComputationAwareKalman.Σ(dgmp, k),
            M⁻ₖ,
            ComputationAwareKalman.H(mmod, k),
            ComputationAwareKalman.Λ(mmod, k),
            yₖ,
            abstol = 1e-6,
            reltol = 1e-8,
            # policy=ComputationAwareKalman.RandomGaussianPolicy(
            # 	Random.seed!(492587 + k)
            # ),
            # policy=ComputationAwareKalman.CoordinatePolicy(
            # 	collect(1:length(yₖ))
            # ),
        )

        # Truncate
        M⁺ₖ, Π⁺ₖ = ComputationAwareKalman.truncate(xₖ.M, min_sval = 1e-6)

        push!(fcache; m⁻ = m⁻ₖ, xₖ..., M⁺ = M⁺ₖ, Π⁺ = Π⁺ₖ)

        global mₖ₋₁ = xₖ.m
        global M⁺ₖ₋₁ = M⁺ₖ
    end
end

# ╔═╡ 71fedc42-97d5-494d-b1a0-994daae7e43f
function plot_belief(ts, xs, states, H)
    plot(
        ts,
        xs,
        hcat([H * Statistics.mean(states[k]) for k = 1:length(ts)]...),
        st = :surface,
        colorbar = false,
    )

    plot!(
        ts,
        xs,
        hcat(
            [
                H * Statistics.mean(states[k]) +
                1.96 * sqrt.(diag(H * Statistics.cov(states[k]) * H')) for k = 1:length(ts)
            ]...,
        ),
        st = :surface,
        fa = 0.7,
    )
end;

# ╔═╡ 58dfbaa8-dc8f-431f-bdce-2b5b7036dd1d
plot_belief(
    ts,
    xs_plot,
    [
        ComputationAwareKalman.ConditionalGaussianBelief(
            ComputationAwareKalman.m(fcache, k),
            ComputationAwareKalman.Σ(dgmp, k),
            ComputationAwareKalman.M(fcache, k),
        ) for k = 1:length(ts)
    ],
    H_plot,
)

# ╔═╡ a64dd0f3-2b08-413d-891e-25a0977c417c
plot_belief(
    ts_plot,
    xs_plot,
    [ComputationAwareKalman.interpolate(dgmp, fcache, t) for t in ts_plot],
    H_plot,
)

# ╔═╡ be5092fa-a620-4d31-9fe0-fdb9e5282565
scache = ComputationAwareKalman.smooth(dgmp, fcache, truncate_kwargs = (min_sval = 1e-3,));

# ╔═╡ efd19ba3-5201-409c-bfd9-add678240d5c
plot_belief(
    ts,
    xs_plot,
    [
        ComputationAwareKalman.ConditionalGaussianBelief(
            ComputationAwareKalman.mˢ(scache, k),
            ComputationAwareKalman.Σ(dgmp, k),
            ComputationAwareKalman.Mˢ(scache, k),
        ) for k = 1:length(ts)
    ],
    H_plot,
)

# ╔═╡ 8885666a-95df-474c-8502-c3f5a80a9548
plot_belief(
    ts_plot,
    xs_plot,
    [ComputationAwareKalman.interpolate(dgmp, fcache, scache, t) for t in ts_plot],
    H_plot,
)

# ╔═╡ 5beea619-b41f-468d-8ee3-ce319c3357c2
rng = Random.seed!(0);

# ╔═╡ c89fa42f-f93e-4c3e-9130-cd58dd477c72
plot(
    ts_plot,
    xs_plot,
    reshape(rand(rng, p_ftxs(txs_plot)), length(xs_plot), length(ts_plot)),
    st = :surface,
)

# ╔═╡ d1b6d44a-bdd2-4c92-b037-72170631703c
plot(
    ts,
    xs_plot,
    H_plot * hcat(rand(rng, dgmp, mmod, RowVecs(ys), fcache)...),
    st = :surface,
)

# ╔═╡ 30937aca-6764-46ab-bed3-faf0e7d96f1e
plot(
    ts_plot,
    xs_plot,
    H_plot * hcat(rand(rng, dgmp, mmod, RowVecs(ys), fcache, ts_plot)...),
    st = :surface,
)

# ╔═╡ Cell order:
# ╠═3438fb37-dab7-4bea-bf97-e1b087918e98
# ╠═fb31863e-9027-11ed-066a-7fc870cb91db
# ╠═e79effe2-213b-49ed-9ce9-0b4d48a2f190
# ╠═bbf87054-1e15-4fcb-a65f-7ec42018eeb6
# ╠═ca0bc9de-32e8-4fcc-acd6-46c5b522c99e
# ╠═4633f1f9-97ac-4e03-ac1e-3a87f90fb879
# ╠═3f0793a0-e993-4e1d-bf77-5c2940832ab3
# ╠═cba9c673-aa11-4226-ad73-d36133f78fde
# ╠═93346804-c558-49d5-9872-c11d9ed6e258
# ╠═abe83b9d-f73f-4d54-b7ac-f8d953dfc5ad
# ╠═e6cbe302-a906-4dcf-9084-868c78343dcf
# ╠═3e129fc5-a5b9-4d74-9470-5c83ee831c59
# ╠═7d149836-b07b-468a-bdb0-8d8cae45ae23
# ╠═71fedc42-97d5-494d-b1a0-994daae7e43f
# ╠═58dfbaa8-dc8f-431f-bdce-2b5b7036dd1d
# ╠═a64dd0f3-2b08-413d-891e-25a0977c417c
# ╠═be5092fa-a620-4d31-9fe0-fdb9e5282565
# ╠═efd19ba3-5201-409c-bfd9-add678240d5c
# ╠═8885666a-95df-474c-8502-c3f5a80a9548
# ╠═5beea619-b41f-468d-8ee3-ce319c3357c2
# ╠═c89fa42f-f93e-4c3e-9130-cd58dd477c72
# ╠═d1b6d44a-bdd2-4c92-b037-72170631703c
# ╠═30937aca-6764-46ab-bed3-faf0e7d96f1e
