## support / pmf for Base types ##############################################

"""
    support(distribution)

Return a collection with unique elements containing at least all
the possible outcomes from `distribution`. For numbers and some other types,
it is sorted.

# Examples
```jldoctest
julia> support(1:3)
1:3

julia> support([4, 2, 4])
2:4

julia> support(([3], [1]))
2-element Array{Array{Int64,1},1}:
 [1]
 [3]
```
"""
function support end

issortable(::Type) = false

for T in (Real,AbstractChar,String)
    @eval begin
        issortable(::Type{<:$T}) = true
        issortable(::Type{<:Tuple{Vararg{$T}}}) = true
        issortable(::Type{<:AbstractArray{<:$T}}) = true
    end
end

support(x::Union{AbstractArray{<:Integer},Tuple{Vararg{Integer}}}) =
    UnitRange(extrema(x)...)

support(x::Union{AbstractArray,Tuple})  =
    issortable(typeof(x)) ?
        unique!(sort!(vec(collect(x)))) :
        unique!(vec(collect(x)))


"""
    pmf(distribution, x)

Return the evaluation of the probability mass function of a (discrete)
`distribution` at `x`.

# Examples
```jldoctest
julia> [pmf([1, 2, 4, 1], x) for x=0:4]
5-element Array{Float64,1}:
 0.0
 0.5
 0.25
 0.0
 0.25
```
"""
pmf(distribution, x)

function pmf(A::Union{AbstractArray,Tuple}, x)
    x ∈ A || return 0.0
    c = count(isequal(x), A)
    c * 1.0 / length(A)
end

pmf(A::AbstractRange, x) = Float64(x ∈ A) / length(A)

function pmf(A::Union{AbstractArray,Tuple})
    d = Dict{eltype(A),Float64}()
    r = 1/length(A)
    for x in A
        p = get(d, x, 0.0)
        d[x] = p + r
    end
    PMF(A, d)
end


## PMF #######################################################################

"""
    pmf(distribution)

Return the probability mass function of a (discrete) `distribution`
as a `PMF` object, which caches evaluations.

# Examples
```jldoctest
julia> v = [1, 2, 3, 1]; f = pmf(v);

julia> f(1) # result computed from pmf(v, 1) and cached in f
0.5

julia> f # when displayed, all values are computed and cached
pmf for [1, 2, 3, 1] with support of length 3:
  1 => 0.5
  2 => 0.25
  3 => 0.25
```
"""
pmf(d) = PMF(d)

# a struct to cache values of pmf
# function, and map interface mostly for printing purposes
mutable struct PMF{T,D} <: AbstractDict{T,Float64}
    d::D
    pmf::Dict{T,Float64}
    cached::Bool                      # all values cached
    support::Union{Nothing,Vector{T}} # !== nothing when keys is sorted
end

function PMF(d)
    T = gentype(d)
    PMF{T,typeof(d)}(d, Dict{T,Float64}(), false, nothing)
end

function PMF(d, probas::Dict{T,Float64}) where T
    gentype(d) == T ||
        throw(ArgumentError("distribution and dictionary are incompatibles"))

    f = PMF(d, probas, true, nothing)
    if issortable(T)
        f.support = sort!(collect(keys(probas)))
    end
    f
end

pmf(d::PMF) = d

function cacheall!(f::PMF)
    if !f.cached
        for x in support(f.d)
            f(x)
        end
        f.cached = true
        if issortable(keytype(f))
           f.support = sort!(collect(keys(f.pmf)))
        end
    end
    f
end

function (f::PMF)(x)
    if haskey(f.pmf, x)
        f.pmf[x]
    elseif f.cached
        0.0
    else
        p = pmf(f.d, x)
        if p != 0.0
            f.pmf[x] = p
        end
        p
    end
end

support(f::PMF) = keys(f)


### Dict interface (for printing)

function Base.keys(f::PMF)
    cacheall!(f)
    f.support !== nothing ?
        f.support :
        keys(f.pmf)
end

function Base.values(f::PMF)
    cacheall!(f)
    f.support !== nothing ?
        (f.pmf[x] for x in f.support) :
        values(f.pmf)
end

Base.getindex(f::PMF, x) = cacheall!(f).pmf[x]
Base.length(f::PMF) = length(cacheall!(f).pmf)

Base.summary(io::IO, f::PMF) =
    print(io, "pmf for ", f.d, " with support of length ", length(f))

function Base.iterate(f::PMF, iter=iterate(keys(cacheall!(f))))
    iter === nothing && return nothing
    (iter[1] => f.pmf[iter[1]]), iterate(keys(f), iter[2])
end


## sampling

# hack: PMF <: AbstractDict for printing purposes, but we want it to be
# also a distribution, so we need to specialize gentype

Sampler(::Type{RNG}, f::PMF, n::Repetition) where {RNG<:AbstractRNG} =
    Sampler(RNG, f.d, n)

Random.gentype(::Type{<:PMF{T}}) where {T} = T
