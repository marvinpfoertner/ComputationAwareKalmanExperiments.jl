using Makie

function axis_sphere(
    args...;
    aspect = :equal,
    azimuth = 0.274 * π,
    protrusions = (0, 0, 0, 20),
    kwargs...
)
    ax = Axis3(
        args...;
        aspect,
        azimuth,
        protrusions,
        kwargs...,
    )

    hidedecorations!(ax)
    hidespines!(ax)

    return ax
end

function plot_heatmap_sphere!(ax, era5::ERA5, values, args...; kwargs...)
    plot_heatmap_sphere!(ax, era5, Observable(values), args...; kwargs...)
end

function plot_heatmap_sphere!(ax, era5::ERA5, values::Observable{<:AbstractVector}, args...; kwargs...)
    plot_heatmap_sphere!(
        ax,
        era5,
        map(v -> reshape(v, length(era5.λs), length(era5.θs)), values),
        args...;
        kwargs...,
    )
end

function plot_heatmap_sphere!(ax, era5::ERA5, values::Observable, args...; kwargs...)
    surface!(
        ax,
        reverse(vcat(era5.xyzs[:, :, 1], era5.xyzs[1, :, 1]'), dims=1),
        reverse(vcat(era5.xyzs[:, :, 2], era5.xyzs[1, :, 2]'), dims=1),
        reverse(vcat(era5.xyzs[:, :, 3], era5.xyzs[1, :, 3]'), dims=1),
        args...;
        color=map(v -> reverse(vcat(v, v[1, :]'), dims=1), values),
        kwargs...,
    )
end
