seeds = 1:5
ranks = 2 .^ (0:3:9)

configs = (cakf_caks = [(rank = rank,) for rank in ranks],)

#####################
# Common parameters #
#####################

data_seed_offset = 2345

N_t = 100
ts_train_idcs = [5, 9, 12, 14, 19, 25, 46, 53, 68, 99]
# sort!(randperm(N_t)[1:10])
N_x_train = 40
