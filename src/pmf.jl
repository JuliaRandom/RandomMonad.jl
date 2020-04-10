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

pmf_count(distribution) = 1.0

function pmf(A::Union{AbstractArray,Tuple}, x)
    x ∈ A || return 0.0
    c = count(isequal(x), A)
    c * 1.0 / length(A)
end

pmf(A::AbstractRange, x) = Float64(x ∈ A) / length(A)

function pmf(A::Union{AbstractArray,Tuple}; normalized::Bool=true)
    d = Dict{eltype(A),Float64}()
    r = normalized ? 1/length(A) : 1.0
    for x in A
        p = get(d, x, 0.0)
        d[x] = p + r
    end
    PMF(A, d, normalized=normalized, count=length(A))
end


## PMF #######################################################################

"""
    pmf(distribution; normalized::Bool=true)

Return the probability mass function of a (discrete) `distribution`
as a `PMF` object, which caches evaluations.
If `normalized` is `false`, the evaluations of the function must be
interpreted not as probabilities but as "weigths", which are computed
according to the specificities of the `distribution`
(the sum of the weights somehow correspond to the size of the sample
space with "multiplicities"). Not all distributions support the
`normalized` keyword.

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

julia> pmf(v, normalized=false)
non-normalized pmf for [1, 2, 3, 1] with support of length 3:
  1 => 2.0
  2 => 1.0
  3 => 1.0
```
"""
pmf(d) = PMF(d)

# a struct to cache values of pmf
# function, and map interface mostly for printing purposes
mutable struct PMF{T,D} <: AbstractDict{T,Float64}
    d::D
    pmf::Dict{T,Float64}
    cached::Bool                      # all values cached
    count::Float64                    # sample count with multiplicities
    support::Union{Nothing,Vector{T}} # !== nothing when keys is sorted
end

function PMF(d)
    T = gentype(d)
    PMF{T,typeof(d)}(d, Dict{T,Float64}(), false, -pmf_count(d), nothing)
end

function PMF(d, probas::Dict{T,Float64};
             normalized::Bool=true, count=nothing) where T
    gentype(d) == T ||
        throw(ArgumentError("distribution and dictionary are incompatibles"))

    if count === nothing
        if normalized
            count = 1.0
        else
            count = sum(values(probas))
        end
    else
        count > 0 || throw(ArgumentError("count must be > 0"))
    end

    resort!(PMF(d, probas, true, Float64(normalized ? -count : count),
                nothing))
end

pmf(f::PMF) = f

isnormalized(f::PMF) = f.count < 0.0 || f.count == 1.0 # use isapprox ?

"""
    normalize!(f::PMF)
    denormalize!(f::PMF)

Make the evaluations of `f` normalized (their sum is approximately 1.0) or
denormalized.

# Examples
```julia-repl
julia> f = pmf(rand(1:3, 3), normalized=false)
Non-normalized pmf for [3, 3, 1] with support of length 2:
  1 => 1.0
  3 => 2.0

julia> normalize!(f)
pmf for [3, 3, 1] with support of length 2:
  1 => 0.333333
  3 => 0.666667

julia> denormalize!(f)
Non-normalized pmf for [3, 3, 1] with support of length 2:
  1 => 1.0
  3 => 2.0
```
"""
normalize!, denormalize!

function normalize!(f::PMF)
    cacheall!(f)
    isnormalized(f) && return f
    replace!(f.pmf) do x
        first(x) => last(x) / f.count
    end
    f.count = -f.count
    f
end

function denormalize!(f::PMF)
    cacheall!(f)
    f.count > 0 && return f
    f.count = -f.count
    replace!(f.pmf) do x
        first(x) => last(x) * f.count
    end
    f
end

function Base.filter!(g, f::PMF)
    cacheall!(f)
    # TODO: optimize (avoid two passes over f.pmf)
    s = 0.0
    replace!(f.pmf) do kv
        if g(kv[1])
            kv
        else
            s += kv[2]
            kv[1] => 0.0
        end
    end
    filter!(x -> x[2] != 0.0, f.pmf)
    s == 0.0 && return f # nothing happened
    if f.count > 0 # non-normalized
        f.count -= s
    else # normalized
        f.count = 1 - s # make non-normalized
        # this allows to keep probability-like values, but which don't
        # sum up to 1.0, so f can't be considered normalized anymore
        # old value of count is totally discarded
    end
    resort!(f)
end

function cacheall!(f::PMF)
    if !f.cached
        for x in support(f.d)
            f(x)
        end
        f.cached = true
        resort!(f)
    end
    f
end

function resort!(f::PMF)
    @assert f.cached
    if issortable(keytype(f))
        f.support = sort!(collect(keys(f.pmf)))
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

function Base.summary(io::IO, f::PMF)
    isnormalized(f) ||
        print(io, "non-normalized ")
    print(io, "pmf for ", f.d, " with support of length ", length(f))
end

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
