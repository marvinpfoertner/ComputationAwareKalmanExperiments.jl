julia --project=@. experiments/era5/01_filter.jl $1
julia --project=@. experiments/era5/02_smoother.jl $1
julia --project=@. experiments/era5/03_metrics_ongrid.jl $1
