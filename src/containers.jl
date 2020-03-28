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
