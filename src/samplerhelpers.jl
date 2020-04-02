## sampler ###################################################################

# allows to call `Sampler` only when the the arg isn't a Sampler itself

sampler(::Type{RNG}, X, n::Repetition) where {RNG<:AbstractRNG} =
    Sampler(RNG, X, n)

sampler(::Type{RNG}, ::Type{X}, n::Repetition) where {RNG<:AbstractRNG,X} =
    Sampler(RNG, X, n)

sampler(::Type{RNG}, X::Sampler, n::Repetition) where {RNG<:AbstractRNG} = X

sampler(rng::AbstractRNG, X, n::Repetition) = sampler(typeof(rng), X, n)


## SamplerTrivial, SamplerSimple, SamplerTag #################################

# we redefine (copy-paste) SamplerTrivial, SamplerSimple, SamplerTag from
# Random so that they inherit from SamplerReset

struct SamplerTrivial{T,E} <: SamplerReset{E}
    self::T
end

"""
    SamplerTrivial(x)

Create a sampler that just wraps the given value `x`. This is the default fall-back for
values.
The `eltype` of this sampler is equal to `eltype(x)`.

The recommended use case is sampling from values without precomputed data.
"""
SamplerTrivial(x::T) where {T} = SamplerTrivial{T,gentype(T)}(x)

Sampler(::Type{<:AbstractRNG}, x::Distribution, ::Repetition) = SamplerTrivial(x)

Base.getindex(sp::SamplerTrivial) = sp.self

# simple sampler carrying data (which can be anything)
struct SamplerSimple{T,S,E} <: SamplerReset{E}
    self::T
    data::S
end

"""
    SamplerSimple(x, data)

Create a sampler that wraps the given value `x` and the `data`.
The `eltype` of this sampler is equal to `eltype(x)`.

The recommended use case is sampling from values with precomputed data.
"""
SamplerSimple(x::T, data::S) where {T,S} = SamplerSimple{T,S,gentype(T)}(x, data)

Base.getindex(sp::SamplerSimple) = sp.self

# simple sampler carrying a (type) tag T and data
struct SamplerTag{T,S,E} <: SamplerReset{E}
    data::S
    SamplerTag{T}(s::S) where {T,S} = new{T,S,gentype(T)}(s)
end
