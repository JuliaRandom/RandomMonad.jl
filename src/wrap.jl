## Uniform

abstract type Uniform{T} <: Distribution{T} end


struct UniformType{T} <: Uniform{T} end

Uniform(::Type{T}) where {T} = UniformType{T}()

Base.getindex(::UniformType{T}) where {T} = T

struct UniformWrap{T,E} <: Uniform{E}
    val::T
end

Uniform(x::T) where {T} = Uniform(x, isuniform(T))

Uniform(x::T, ::Val{true}) where {T} = UniformWrap{T,gentype(T)}(x)

Base.getindex(x::UniformWrap) = x.val


const ImplicitUniform = Union{AbstractArray,AbstractDict,AbstractSet,
                              AbstractString,Tuple}

isuniform(::Type{T}) where {T} = Val(false)
isuniform(::Type{T}) where {T<:ImplicitUniform} = Val(true)


## Wrap


struct Wrap{T,E} <: Distribution{E}
    val::T
end

Wrap(x::T) where {T} = Wrap{T,gentype(T)}(x)

Base.getindex(x::Wrap) = x.val


## wrap

wrap(::Type{T}) where {T} = Uniform(T)

wrap(x::T) where {T} = wrap(x, isuniform(T))

wrap(x::T, ::Val{true}) where {T} = Uniform(x)
wrap(x::T, ::Val{false}) where {T} = Wrap(x)

wrap(x::Distribution) = x
wrap(x::Sampler) = x


## sampling

Sampler(RNG::Type{<:AbstractRNG},
        d::Union{UniformWrap,UniformType,Wrap}, n::Repetition) =
            Sampler(RNG, d[], n)
