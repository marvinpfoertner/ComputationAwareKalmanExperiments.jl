function discretize_truncate(
    stsgmp::ComputationAwareKalman.SpaceTimeSeparableGaussMarkovProcess,
    xs;
    rank::Integer,
)
    rank % ComputationAwareKalman.statedim(stsgmp) == 0 ||
        error("`rank` must be a multiple of state space dimension")

    # Low-rank approximation of left square root of spatial covariance matrix
    spatial_cov_mat = ComputationAwareKalman.covariance_matrix(stsgmp.spatial_cov_fn, xs)

    eigvals, eigvecs = eigen(spatial_cov_mat)

    spatial_rank = rank ÷ ComputationAwareKalman.statedim(stsgmp)
    lr_lsqrt_spatial_cov_mat =
        eigvecs[:, end-spatial_rank+1:end] *
        Diagonal(sqrt.(eigvals[end-spatial_rank+1:end]))

    return ComputationAwareKalmanExperiments.KroneckerGaussMarkovProcess(
        stsgmp.tgmp,
        ComputationAwareKalman.mean_vector(stsgmp.spatial_mean_fn, xs),
        lr_lsqrt_spatial_cov_mat * lr_lsqrt_spatial_cov_mat',
        lr_lsqrt_spatial_cov_mat,
    )
end

function discretize_truncate_lanczos(
    stsgmp::ComputationAwareKalman.SpaceTimeSeparableGaussMarkovProcess,
    xs;
    rng::Random.AbstractRNG,
    rank::Integer,
)
    rank % ComputationAwareKalman.statedim(stsgmp) == 0 ||
        error("`rank` must be a multiple of state space dimension")

    # Low-rank approximation of left square root of spatial covariance matrix
    spatial_cov_mat = ComputationAwareKalman.covariance_matrix(stsgmp.spatial_cov_fn, xs)

    spatial_rank = rank ÷ ComputationAwareKalman.statedim(stsgmp)

    eigvals, eigvecs, _ = KrylovKit.eigsolve(
        x -> spatial_cov_mat * x,
        randn(rng, size(spatial_cov_mat, 2)),
        spatial_rank,
        :LM;
        krylovdim = max(KrylovDefaults.krylovdim, spatial_rank),
        orth = KrylovKit.ClassicalGramSchmidt2(),
        issymmetric = true,
    )

    lr_lsqrt_spatial_cov_mat = hcat(eigvecs...) * Diagonal(sqrt.(eigvals))

    return ComputationAwareKalmanExperiments.KroneckerGaussMarkovProcess(
        stsgmp.tgmp,
        ComputationAwareKalman.mean_vector(stsgmp.spatial_mean_fn, xs),
        lr_lsqrt_spatial_cov_mat * lr_lsqrt_spatial_cov_mat',
        lr_lsqrt_spatial_cov_mat,
    )
end
