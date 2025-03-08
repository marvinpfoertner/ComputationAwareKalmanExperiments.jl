using CUDA

struct WrappedCuArray{T,N,TcuA<:CuArray{T, N}} <: AbstractArray{T,N}
    cuA::TcuA
end

# function WrappedCuArray(cuA::CuArray{T, N}) where {T, N}
#     return WrappedCuArray{T, N, typeof(cuA)}(cuA)
# end

Base.size(A::WrappedCuArray) = size(A.cuA)

function Base.:*(A::WrappedCuArray, V::Array)
    cuV = CuArray(V)
    cuY = A.cuA * cuV
    return Array(cuY)
end

function LinearAlgebra.mul!(Y::AbstractMatrix, A::WrappedCuArray, V::AbstractMatrix)
    cuV = CuArray(V)
    cuY = A.cuA * cuV
    copyto!(Y, cuY)
    return Y
end
