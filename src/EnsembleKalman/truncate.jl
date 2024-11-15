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
