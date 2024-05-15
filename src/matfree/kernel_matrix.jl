struct KernelMatrix{T<:Real,TK,TX₁<:AbstractVector,TX₂<:AbstractVector} <: AbstractMatrix{T}
    K::TK
    X₁::TX₁
    X₂::TX₂
end

function KernelMatrix{T}(K::TK, X₁::TX₁, X₂::TX₂=X₁) where {T,TK,TX₁,TX₂}
    return KernelMatrix{T,TK,TX₁,TX₂}(K, X₁, X₂)
end

Base.size(Kₓ₁ₓ₂::KernelMatrix) = (length(Kₓ₁ₓ₂.X₁), length(Kₓ₁ₓ₂.X₂))
Base.IndexStyle(::Type{<:KernelMatrix}) = IndexCartesian()
Base.getindex(Kₓ₁ₓ₂::KernelMatrix, i::Int, j::Int) = Kₓ₁ₓ₂.K(Kₓ₁ₓ₂.X₁[i], Kₓ₁ₓ₂.X₂[j])

LinearAlgebra.adjoint(Kₓ₁ₓ₂::KernelMatrix{T}) where {T} =
    KernelMatrix{T}(Kₓ₁ₓ₂.K, Kₓ₁ₓ₂.X₂, Kₓ₁ₓ₂.X₁)
LinearAlgebra.transpose(Kₓ₁ₓ₂::KernelMatrix{T}) where {T} =
    KernelMatrix{T}(Kₓ₁ₓ₂.K, Kₓ₁ₓ₂.X₂, Kₓ₁ₓ₂.X₁)

function ComputationAwareKalman.covariance_matrix(K, X₁, X₂=X₁)
    return KernelMatrix{Float64}(K, X₁, X₂)
end

#############################################
# GPU KernelMatrix-{Vector,Matrix} Products #
#############################################

using Adapt

Adapt.adapt_structure(to, Kₓ₁ₓ₂::KernelMatrix{T}) where {T} = KernelMatrix{T}(
    adapt(to, Kₓ₁ₓ₂.K),
    adapt(to, Kₓ₁ₓ₂.X₁),
    adapt(to, Kₓ₁ₓ₂.X₂)
)

using CUDA
using CUDA: i32

function kernelmat_matmul_kernel!(
    Y::CuDeviceMatrix, K, X₁::CuDeviceVector, X₂::CuDeviceVector, V::CuDeviceMatrix,
)
    i = (blockIdx().x - 1i32) * blockDim().x + threadIdx().x
    j = (blockIdx().y - 1i32) * blockDim().y + threadIdx().y

    if i <= size(Y, 1) && j <= size(Y, 2)
        res = zero(eltype(Y))

        for k in 1i32:size(V, 1)
            res += K(X₁[i], X₂[k]) * V[k, j]
        end

        Y[i, j] = res
    end

    return nothing
end

function LinearAlgebra.mul!(
    Y_d::CuMatrix,
    Kₓ₁ₓ₂_d::ComputationAwareKalmanExperiments.KernelMatrix{<:Real,<:Any,<:CuVector,<:CuVector},
    V_d::CuMatrix,
)
    nthreads_x = 16
    nthreads_y = 16
    nblocks_x = ceil(Int32, size(Y_d, 1) / nthreads_x)
    nblocks_y = ceil(Int32, size(Y_d, 2) / nthreads_y)

    CUDA.@sync begin
        @cuda threads=(nthreads_x, nthreads_y) blocks=(nblocks_x, nblocks_y) kernelmat_matmul_kernel!(Y_d, Kₓ₁ₓ₂_d.K, Kₓ₁ₓ₂_d.X₁, Kₓ₁ₓ₂_d.X₂, V_d)
    end

    return Y_d
end

function LinearAlgebra.mul!(
    Y::AbstractMatrix{ETy},
    Kₓ₁ₓ₂_d::ComputationAwareKalmanExperiments.KernelMatrix{<:Real,<:Any,<:CuVector,<:CuVector},
    V::AbstractMatrix,
) where {ETy}
    Y_d = CuMatrix{ETy}(undef, size(Y))
    V_d = CuMatrix(V)

    mul!(Y_d, Kₓ₁ₓ₂_d, V_d)

    copyto!(Y, Array(Y_d))  # TODO: Fix this

    return Y
end

function Base.:*(
    Kₓ₁ₓ₂_d::ComputationAwareKalmanExperiments.KernelMatrix{<:Real,<:Any,<:CuVector,<:CuVector},
    V::AbstractMatrix,
)
    Y_d = CuMatrix{promote_type(eltype(Kₓ₁ₓ₂_d), eltype(V))}(undef, size(Kₓ₁ₓ₂_d.X₁, 1), size(V, 2))
    V_d = CuMatrix(V)

    mul!(Y_d, Kₓ₁ₓ₂_d, V_d)

    return Array(Y_d)
end

function kernelmat_matvec_kernel!(
    y::CuDeviceVector, K, X₁::CuDeviceVector, X₂::CuDeviceVector, v::CuDeviceVector,
)
    i = (blockIdx().x - 1i32) * blockDim().x + threadIdx().x

    if i <= size(y, 1)
        res = zero(eltype(y))

        for k in 1i32:size(v, 1)
            res += K(X₁[i], X₂[k]) * v[k]
        end

        y[i] = res
    end

    return nothing
end

function LinearAlgebra.mul!(
    y_d::CuVector,
    Kₓ₁ₓ₂_d::ComputationAwareKalmanExperiments.KernelMatrix{<:Real,<:Any,<:CuVector,<:CuVector},
    v_d::CuVector,
)
    nthreads = 16 * 16
    nblocks = ceil(Int32, length(y_d) / nthreads)

    CUDA.@sync begin
        @cuda threads=nthreads blocks=nblocks kernelmat_matvec_kernel!(y_d, Kₓ₁ₓ₂_d.K, Kₓ₁ₓ₂_d.X₁, Kₓ₁ₓ₂_d.X₂, v_d)
    end

    return y_d
end

function LinearAlgebra.mul!(
    y::AbstractVector{ETy},
    Kₓ₁ₓ₂_d::ComputationAwareKalmanExperiments.KernelMatrix{<:Real,<:Any,<:CuVector,<:CuVector},
    v::AbstractVector,
) where {ETy}
    y_d = CuVector{ETy}(undef, size(y))
    v_d = CuVector(v)

    mul!(y_d, Kₓ₁ₓ₂_d, v_d)

    copyto!(y, Array(y_d))  # TODO: Fix this

    return y
end

function Base.:*(
    Kₓ₁ₓ₂_d::ComputationAwareKalmanExperiments.KernelMatrix{<:Real,<:Any,<:CuVector,<:CuVector},
    v::AbstractVector,
)
    y_d = CuVector{promote_type(eltype(Kₓ₁ₓ₂_d), eltype(v))}(undef, size(Kₓ₁ₓ₂_d.X₁, 1))
    v_d = CuVector(v)

    mul!(y_d, Kₓ₁ₓ₂_d, v_d)

    return Array(y_d)
end

Base.:*(V::AbstractMatrix, Kₓ₁ₓ₂_d::ComputationAwareKalmanExperiments.KernelMatrix{<:Real,<:Any,<:CuVector,<:CuVector}) = (Kₓ₁ₓ₂_d' * V')'

# TODO: The following can be implemented more efficiently!
function LinearAlgebra.diag(Kₓ₁ₓ₂::KernelMatrix{<:Any,<:Any,<:CuVector,<:CuVector})
    return map(Kₓ₁ₓ₂.K, Vector(Kₓ₁ₓ₂.X₁), Vector(Kₓ₁ₓ₂.X₂))
end
