using GLMakie

include("common.jl")

rng = Random.seed!(2345)

ys_test = rand(rng, dgmp, ts)

ys_train = [rand(rng, mmod, k, y_test) for (k, y_test) in enumerate(ys_test[ts_train_idcs])]

jldsave(data_path, ys_test = ys_test, ys_train = ys_train)
