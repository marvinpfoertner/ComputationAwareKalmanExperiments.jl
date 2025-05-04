using NCDatasets

const r_earth = 6371.0  # km
const C_earth = 2.0 * π * r_earth  # km

##########################################################################################
# (Downsampled) ERA5 Dataset #############################################################
##########################################################################################

struct ERA5{Ttsidcs}
    ds_path::String

    ts::Vector{Float64}

    t_idcs::Ttsidcs

    λs::Vector{Float64}
    θs::Vector{Float64}

    step_λ::Int
    step_θ::Int

    xyzs::Array{Float64,3}

    λθs::Vector{Tuple{Float64,Float64}}
end

const Nλ_total::Int = 2 * 2 * 2 * 2 * 2 * 3 * 3 * 5  # = 1440
#                    |_____| |_____________________|
#                      = 4           = 360

const Nθ_total::Int = 2 * 2 * 2 * 2 * 3 * 3 * 5 + 1  # = 721
#                    |_____| |_________________|
#                      = 4          = 180

function ERA5(ds_path::String; t_idcs = 1:7*24, step_λ::Integer = 1, step_θ::Integer = 1)
    isinteger(Nλ_total // step_λ) ||
        throw(DimensionMismatch("Nλ_total is not divisible by step_λ"))
    isinteger((Nθ_total - 1) // step_θ) ||
        throw(DimensionMismatch("Nθ_total - 1 is not divisible by step_θ"))

    ts, λs, θs = NCDataset(ds_path, "r") do ds
        @assert Nλ_total == length(ds["longitude"])
        @assert Nθ_total == length(ds["latitude"])

        ts = ds["valid_time"][t_idcs]

        return (
            Dates.value.(Dates.Hour.(ts .- ts[1])) / 24.0,  # days
            ds["longitude"][1:step_λ:end] .* (π / 180.0),  # rad
            ds["latitude"][1:step_θ:end] .* (π / 180.0),  # rad
        )
    end

    xyzs = reshape(
        stack(
            [
                ComputationAwareKalmanExperiments.gcs_to_cartesian(λ, θ, r = r_earth)
                for θ in θs for λ in λs
            ],
            dims = 1,
        ),
        (length(λs), length(θs), 3),
    )

    λθs = Tuple{Float64,Float64}[(λ, θ) for θ in θs for λ in λs]

    return ERA5(ds_path, ts, t_idcs, λs, θs, Int(step_λ), Int(step_θ), xyzs, λθs)
end

function with_ds(f, era5::ERA5)
    return NCDataset(f, era5.ds_path, "r")
end

function T₂ₘs(era5::ERA5, ds::NCDataset, k::Integer)
    return convert(
        Matrix{Float64},
        ds["t2m"][1:era5.step_λ:end, 1:era5.step_θ:end, era5.t_idcs[k]],
    ) .- 273.15  # °C
end

##########################################################################################
# Train-Test Split of (Downsampled) ERA5 Dataset #########################################
##########################################################################################

struct ERA5TrainTestSplit{Tera5<:ERA5}
    era5::Tera5

    t_idcs_train::Vector{Int}
    t_idcs_test::Vector{Int}

    λθ_idcs_train::Vector{Int}
    λθ_idcs_test::Vector{Int}
end

function ERA5TrainTestSplit(era5::ERA5, t_idcs_test::Vector{Int}, λθ_idcs_test::Vector{Int})
    return ERA5TrainTestSplit(
        era5,
        deleteat!(collect(1:length(era5.ts)), t_idcs_test),
        t_idcs_test,
        deleteat!(collect(1:length(era5.λθs)), λθ_idcs_test),
        λθ_idcs_test,
    )
end

function ERA5TrainTestSplit(
    era5::ERA5,
    t_idcs_test::Vector{Int},
    step_λ_test::Int,
    step_θ_test::Int,
)
    isinteger(length(era5.λs) // step_λ_test) ||
        throw(DimensionMismatch("λs is not divisible by step_λ_test"))
    isinteger((length(era5.θs) - 1) // step_θ_test) ||
        throw(DimensionMismatch("θs[1:end - 1] is not divisible by step_θ_test"))

    λθ_idcs_test = [
        (i - 1) * length(era5.λs) + j for
        i = (1+step_θ_test):step_θ_test:(length(era5.θs)-1) for
        j = 1:step_λ_test:length(era5.λs)
    ]

    return ERA5TrainTestSplit(era5, t_idcs_test, λθ_idcs_test)
end

function ts_train(era5_split::ERA5TrainTestSplit)
    return era5_split.era5.ts[era5_split.t_idcs_train]
end

function ts_test(era5_split::ERA5TrainTestSplit)
    return era5_split.era5.ts[era5_split.t_idcs_test]
end

function T₂ₘs_train(era5_split::ERA5TrainTestSplit, ds::NCDataset, k_train::Integer)
    return reshape(
        T₂ₘs(era5_split.era5, ds, era5_split.t_idcs_train[k_train]),
        length(era5_split.era5.λθs),
    )[era5_split.λθ_idcs_train]
end

function T₂ₘs_test(era5_split::ERA5TrainTestSplit, ds::NCDataset, k_train::Integer)
    return reshape(
        T₂ₘs(era5_split.era5, ds, era5_split.t_idcs_train[k_train]),
        length(era5_split.era5.λθs),
    )[era5_split.λθ_idcs_test]
end

function T₂ₘ_train_mean(era5_split::ERA5TrainTestSplit)
    res = 0.0

    with_ds(era5_split.era5) do ds
        for k_train = 1:length(era5_split.t_idcs_train)
            res += mean(T₂ₘs_train(era5_split, ds, k_train))
        end
    end

    return res / length(era5_split.t_idcs_train)
end
