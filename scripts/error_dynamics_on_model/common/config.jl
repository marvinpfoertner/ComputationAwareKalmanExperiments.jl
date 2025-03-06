seeds = 1:5
ranks = 2 .^ (0:3:9)

configs = (cakf_caks = [(rank = rank,) for rank in ranks],)

#####################
# Common parameters #
#####################

data_seed_offset = 2345
