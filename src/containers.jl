## Zip

# note: Zip(a, b...) is similar to RandomExtensions.make(Tuple, a, b...)

struct Zip{T,X} <: Distribution{T}
    xs::X

    function Zip(as...)
        T = Tuple{map(gentype, as)...}
        xs = map(a -> a isa DataType ? Uniform(a) : a, as)
        new{T,typeof(xs)}(xs)
    end
end

rand(rng::AbstractRNG, sp::SamplerTrivial{<:Zip{T}}) where {T} =
    map(x -> rand(rng, x), sp[].xs)::T

Sampler(::Type{RNG}, z::Zip, n::Val{Inf}) where {RNG<:AbstractRNG} =
    SamplerTag{typeof(z)}(map(x -> Sampler(RNG, x, n), z.xs))

function reset!(sp::SamplerTag{<:Zip}, n...)
    foreach(sp.data) do x
        reset!(x, n...)
    end
    sp
end

rand(rng::AbstractRNG, sp::SamplerTag{<:Zip{T}}) where {T} =
    map(x -> rand(rng, x), sp.data)::T


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
    SamplerTag{typeof(f)}((x    = Sampler(RNG, f.x, Val(Inf)),
                           dims = f.dims))

rand(rng::AbstractRNG, sp::SamplerTag{<:Fill}) =
    rand!(rng, gentype(sp)(undef, sp.data.dims), sp, Val(1))

function rand!(rng::AbstractRNG, a::AbstractArray, sp::SamplerTag{<:Fill}, ::Val{N}) where N
    dims = sp.data.dims
    if dims != size(a)
        if length(dims) == 1 && a isa AbstractVector
            resize!(a, dims[1])
        else
            throw(ArgumentError("can not resize destination array"))
        end
    end
    x = reset!(sp.data.x, prod(dims))
    for i in eachindex(a)
        if N == 1
            @inbounds a[i] = rand(rng, x)
        else
            @inbounds rand!(rng, a[i], x, Val(N-1))
        end
    end
    return a
end
