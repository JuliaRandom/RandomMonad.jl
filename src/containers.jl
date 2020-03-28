## Zip

# note: Zip(a, b) is similar to RandomExtensions.make(Tuple, a, b)

struct Zip{T,A,B} <: Distribution{T}
    a::A
    b::B

    Zip(a::A, b::B) where {A,B} =
        new{Tuple{gentype(a),gentype(b)},A,B}(a, b)
end

rand(rng::AbstractRNG, sp::SamplerTrivial{<:Zip{T}}) where {T} =
     (rand(rng, sp[].a), rand(rng, sp[].b))::T

Sampler(RNG::Type{<:AbstractRNG}, m::Zip, n::Val{Inf}) =
    SamplerTag{typeof(m)}((Sampler(RNG, m.a, n),
                           Sampler(RNG, m.b, n)))

rand(rng::AbstractRNG, sp::SamplerTag{<:Zip{T}}) where {T} =
     (rand(rng, sp.data[1]), rand(rng, sp.data[2]))::T


## Fill

struct Fill{X,T,N} <: Distribution{Array{T,N}}
    x::X
    dims::Dims{N}

    Fill(x::X, dims::Dims{N}) where {X,N} = new{X,gentype(x),N}(x, dims)
end

Fill(x, dims::Integer...) where {X} = Fill(x, Dims(dims))

Fill(::Type{X}, dims::Dims{N})    where {X,N} = Fill(Uniform(X), dims)
Fill(::Type{X}, dims::Integer...) where {X}   = Fill(Uniform(X), Dims(dims))


Sampler(RNG::Type{<:AbstractRNG}, f::Fill, n::Repetition) =
    SamplerTag{typeof(f)}((x    = Sampler(RNG, f.x, n),
                           dims = f.dims))

rand(rng::AbstractRNG, sp::SamplerTag{<:Fill}) =
    rand(rng, sp.data.x, sp.data.dims)
