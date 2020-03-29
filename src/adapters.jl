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


## Unique

struct Unique{T,X} <: Distribution{T}
    x::X

    Unique(x::X) where {X} = new{gentype(x),X}(x)
end

Unique(::Type{X}) where {X} = Unique(Uniform(X))

Sampler(RNG::Type{<:AbstractRNG}, u::Unique, n::Val{1}) =
    Sampler(RNG, u.x, n)

Sampler(RNG::Type{<:AbstractRNG}, u::Unique, n::Val{Inf}) =
    SamplerTag{typeof(u)}((x    = Sampler(RNG, u.x, n),
                           seen = Set{gentype(u)}()))

function reset!(sp::SamplerTag{<:Unique}, n=0)
    seen = sp.data.seen
    n > length(seen) && sizehint!(seen, n)
    empty!(seen)
    sp
end

function rand(rng::AbstractRNG, sp::SamplerTag{<:Unique})
    seen = sp.data.seen
    while true
        x = rand(rng, sp.data.x)
        x in seen && continue
        push!(seen, x)
        return x
    end
end


## Fisher-Yates

struct FisherYates{T,N,A} <: Distribution{T}
    a::A

    FisherYates(a::AbstractArray{T,N}) where {T,N} =
        new{T,N,typeof(a)}(a)
end

Sampler(RNG::Type{<:AbstractRNG}, fy::FisherYates, n::Val{1}) =
    Sampler(RNG, fy.a, n)

Sampler(RNG::Type{<:AbstractRNG}, fy::FisherYates, n::Val{Inf}) =
    reset!(SamplerSimple(fy, Vector{Int}(undef, length(fy.a))))

function reset!(sp::SamplerSimple{<:FisherYates}, _=0)
    copy!(sp.data, 1:length(sp[].a))
    sp
end

function rand(rng::AbstractRNG, sp::SamplerSimple{<:FisherYates})
    inds = sp.data
    n = length(inds)
    i = rand(rng, 1:n)
    @inbounds x = inds[i]
    @inbounds inds[i] = inds[end]
    resize!(inds, n-1)
    @inbounds sp[].a[x]
end
