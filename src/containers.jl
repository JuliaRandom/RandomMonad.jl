## variate_size ##############################################################

"""
    variate_size(distribution) :: Dims

Return the `size` of each sample from `distribution`, when it can be
determined in advance. This is used in particular by `Pack`.

# Examples
```julia-repl
julia> variate_size(Normal())
()

julia> variate_size(Fill(Normal(), 2, 3))
(2, 3)

julia> rand(SubSeq(1:9, 0.3), 2)
2-element Array{Array{Int64,1},1}:
 [4, 7]
 [5]

julia> variate_size(SubSeq(1:9, 0.3)) # no pre-determined size
ERROR: MethodError: no method matching variate_size(::SubSeq{Array{Int64,1},UnitRange{Int64}})
[...]
```
"""
function variate_size end

variate_size(d::Distribution{<:Number}) = ()


## Zip #######################################################################

# note: Zip(a, b...) is similar to RandomExtensions.make(Tuple, a, b...)

struct Zip{T,X} <: Distribution{T}
    xs::X

    function Zip(as...)
        T = Tuple{map(gentype, as)...}
        xs = map(a -> a isa DataType ? Uniform(a) : a, as)
        new{T,typeof(xs)}(xs)
    end
end


support(z::Zip) = Iterators.product((support(x) for x in z.xs)...)

function pmf(z::Zip, xs::Tuple)
    length(xs) == length(z.xs) || return 0.0
    prod(pmf(z.xs[i], xs[i]) for i in 1:length(xs))
end

function Base.show(io::IO, z::Zip)
    print(io, "Zip(")
    join(io, z.xs, ", ")
    print(io, ')')
end


### sampling

Sampler(::Type{RNG}, z::Zip, n::Repetition) where {RNG<:AbstractRNG} =
    SamplerTag{typeof(z)}(map(x -> sampler(RNG, x, n), z.xs))

function reset!(sp::SamplerTag{<:Zip}, n...)
    foreach(sp.data) do x
        reset!(x, n...)
    end
    sp
end

rand(rng::AbstractRNG, sp::SamplerTag{<:Zip{T}}) where {T} =
    map(x -> rand(rng, x), sp.data)::T

function rand!(rng::AbstractRNG, t::T,
               sp::SamplerTag{<:Zip{S}},
               ::Val{N}) where N where S where T<:Tuple
    N < 2 && throw(ArgumentError("can not mutate a tuple"))
    fieldcount(T) != fieldcount(S) &&
        throw(ArgumentError("tuples must have same size"))

    for (x, s) in zip(t, sp.data)
        rand!(rng, x, s, Val(N-1))
    end
    t
end


## Fill ######################################################################

struct Fill{X,T,N} <: Distribution{Array{T,N}}
    x::X
    dims::Dims{N}

    Fill(x::X, dims::Dims{N}) where {X,N} = new{X,gentype(x),N}(x, dims)
end

Fill(x, dims::Integer...) where {X} = Fill(x, Dims(dims))

Fill(::Type{X}, dims::Dims{N})    where {X,N} = Fill(Uniform(X), dims)
Fill(::Type{X}, dims::Integer...) where {X}   = Fill(Uniform(X), Dims(dims))

variate_size(f::Fill) = f.dims

Sampler(RNG::Type{<:AbstractRNG}, f::Fill, n::Repetition) =
    SamplerTag{typeof(f)}((x    = sampler(RNG, f.x, Val(Inf)),
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

# TODO: create a lazy fpmf dict which doesn't store all values, but only
# the pmf of f.x, it's then cheap to compute values and keys on the fly
function pmf(f::Fill; normalized=true)
    A = eltype(f)
    xf = pmf(f.x, normalized=normalized)
    fpmf = Dict{A,Float64}()
    n = prod(f.dims)
    # TODO: don't use Iterators.product, which would need "static" n
    for xs in Iterators.product(Iterators.repeated(xf, n)...)
        e = A(undef, f.dims)
        a = copyto!(e, first.(xs))
        @assert !haskey(fpmf, a)
        fpmf[a] = prod(last, xs)
    end
    PMF(f, fpmf, normalized=normalized; count=abs(xf.count)^n)
end

function Base.show(io::IO, f::Fill)
    print(io, "Fill(", f.x, ", ")
    join(io, f.dims, ", ")
    print(io, ')')
end


## Pack ######################################################################

struct Pack{X,T,N} <: Distribution{Array{T,N}}
    x::X
    dims::Dims{N} # (inner_dims..., outer_dims...)
    dim::Int # length of inner_dims

    function Pack(x::X, dims::Dims{N}) where {X,N}
        A = gentype(x)
        inner = variate_size(x)
        dims = (inner..., dims...)
        new{X,eltype(gentype(x)),length(dims)}(x, dims, length(inner))
    end
end

Pack(x, dims::Integer...) where {X} = Pack(x, Dims(dims))

Pack(::Type{X}, dims::Dims{N})    where {X,N} = Pack(Uniform(X), dims)
Pack(::Type{X}, dims::Integer...) where {X}   = Pack(Uniform(X), Dims(dims))

function Base.show(io::IO, p::Pack)
    len = length(variate_size(p.x))
    print(io, "Pack(", p.x, ", ")
    join(io, p.dims[len+1:end], ", ")
    print(io, ')')
end

Sampler(RNG::Type{<:AbstractRNG}, p::Pack, n::Repetition) =
    SamplerTag{typeof(p)}((x    = sampler(RNG, p.x, Val(Inf)),
                           dims = p.dims,
                           dim  = p.dim))

function rand!(rng::AbstractRNG, A::AbstractArray{T,N},
               sp::SamplerTag{Pack{X,T,N}}, ::Val{1}) where {X,T,N}

    size(A) == sp.data.dims || throw(DimensionMismatch(
    "size of destination array ($(size(A))) does not match" *
        "dimensions of Pack ($(sp.data.dims))"))

    inner_idxs = CartesianIndices(sp.data.dims[1:sp.data.dim])
    idxs = CartesianIndices(sp.data.dims[sp.data.dim+1:end])
    x = sp.data.x

    reset!(x, length(idxs))

    if gentype(x) <: AbstractArray
        for idx in idxs
            rand!(rng, view(A, inner_idxs, idx), x, Val(1))
        end
    else
        for idx in idxs # similar to Fill
            @inbounds A[idx] = rand(rng, x)
        end
    end
    A
end

rand(rng::AbstractRNG, sp::SamplerTag{Pack{X,T,N}}) where {X,T,N} =
    rand!(rng, Array{T,N}(undef, sp.data.dims), sp, Val(1))
