struct RestrictionMatrix{Tindices} <: AbstractMatrix{Bool}
    num_cols::Int
    indices::Tindices
end

Base.size(H::RestrictionMatrix) = (length(H.indices), H.num_cols)
Base.IndexStyle(::Type{<:RestrictionMatrix}) = IndexCartesian()
Base.getindex(H::RestrictionMatrix{T}, i::Int, j::Int) where {T} = (j == H.indices[i])

Base.:*(H::RestrictionMatrix, x::AbstractVector) = x[H.indices]

const MulMatTypes = [:AbstractMatrix, :Diagonal]

for TX in MulMatTypes
    @eval function Base.:*(H::RestrictionMatrix, X::$TX)
        return X[H.indices, :]
    end

    @eval function Base.:*(X::$TX, Hᵀ::LinearAlgebra.Adjoint{Bool,<:RestrictionMatrix})
        return X[:, Hᵀ'.indices]
    end
end

Base.:*(H::RestrictionMatrix, Kₓ₁ₓ₂::CovarianceFunctions.Gramian) =
    CovarianceFunctions.Gramian(Kₓ₁ₓ₂.k, Kₓ₁ₓ₂.x[H.indices], Kₓ₁ₓ₂.y)
Base.:*(
    Kₓ₁ₓ₂::CovarianceFunctions.Gramian,
    Hᵀ::LinearAlgebra.Adjoint{Bool,<:RestrictionMatrix},
) = CovarianceFunctions.Gramian(Kₓ₁ₓ₂.k, Kₓ₁ₓ₂.x, Kₓ₁ₓ₂.y[Hᵀ'.indices])

Base.:*(H::RestrictionMatrix, Kₓ₁ₓ₂::KernelMatrix{T}) where {T} =
    KernelMatrix{T}(Kₓ₁ₓ₂.K, Kₓ₁ₓ₂.X₁[H.indices], Kₓ₁ₓ₂.X₂)
Base.:*(
    Kₓ₁ₓ₂::KernelMatrix{T},
    Hᵀ::LinearAlgebra.Adjoint{Bool,<:RestrictionMatrix},
) where {T} = KernelMatrix{T}(Kₓ₁ₓ₂.K, Kₓ₁ₓ₂.X₁, Kₓ₁ₓ₂.X₂[Hᵀ'.indices])

Base.:*(
    H::RestrictionMatrix,
    Kₓ₁ₓ₂::KernelMatrix{T,<:Any,<:CuVector,<:CuVector},
) where {T<:Real} = KernelMatrix{T}(Kₓ₁ₓ₂.K, Kₓ₁ₓ₂.X₁[H.indices], Kₓ₁ₓ₂.X₂)
Base.:*(
    Kₓ₁ₓ₂::KernelMatrix{T,<:Any,<:CuVector,<:CuVector},
    Hᵀ::LinearAlgebra.Adjoint{Bool,<:RestrictionMatrix},
) where {T<:Real} = KernelMatrix{T}(Kₓ₁ₓ₂.K, Kₓ₁ₓ₂.X₁, Kₓ₁ₓ₂.X₂[Hᵀ'.indices])
