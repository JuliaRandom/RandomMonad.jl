## Filter

struct Filter{T,F,D<:Distribution{T}} <: Distribution{T}
    f::F
    d::D
end

Filter(f::F, d) where {F} = Filter(f, Uniform(d))


### sampling

Sampler(RNG::Type{<:AbstractRNG}, d::Filter, n::Repetition) =
    SamplerTag{typeof(d)}((f = d.f,
                           d = Sampler(RNG, d.d, n)))

rand(rng::AbstractRNG, sp::SamplerTag{<:Filter}) =
    while true
        x = rand(rng, sp.data.d)
        sp.data.f(x) && return x
    end


## Map

struct Map{T,F,D} <: Distribution{T}
    f::F
    d::D
end

Map{T}(f::F, d...) where {T,F} = Map{T,F,typeof(d)}(f, d)

function Map(f::F, d...) where {F}
    rt = Base.return_types(f, map(gentype, d))
    T = length(rt) > 1 ? Any : rt[1]
    Map{T}(f, d...)
end


### sampling

# Repetition -> Val(1)
rand(rng::AbstractRNG, sp::SamplerTrivial{<:Map{T}}) where {T} =
    convert(T, sp[].f((rand(rng, d) for d in sp[].d)...))

Sampler(RNG::Type{<:AbstractRNG}, m::Map, n::Val{Inf}) =
    SamplerTag{typeof(m)}((f = m.f,
                           d = map(x -> Sampler(RNG, x, n), m.d)))

rand(rng::AbstractRNG, sp::SamplerTag{<:Map{T}}) where {T} =
    convert(T, sp.data.f((rand(rng, d) for d in sp.data.d)...))
