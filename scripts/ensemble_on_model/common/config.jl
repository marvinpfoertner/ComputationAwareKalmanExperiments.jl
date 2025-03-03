sample_ranks = 2 .^ (1:10)
etkf_lanczos_ranks = 2 .^ (0:10)
cakf_ranks = 2 .^ (0:9)

seeds = 1:5

configs = (
    kf = [(;)],
    srkf = [(;)],
    enkf = [(rank = rank,) for rank in sample_ranks],
    etkf_sample = [(rank = rank,) for rank in sample_ranks],
    etkf_lanczos = [(rank = rank,) for rank in etkf_lanczos_ranks],
    cakf = [(rank = rank,) for rank in cakf_ranks],
)

#####################
# Common parameters #
#####################

data_seed_offset = 2345
