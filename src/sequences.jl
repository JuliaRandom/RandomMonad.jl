## Iterate ###################################################################

"""
    Iterate(iter) :: Distribution{eltype(iter)}

Create a distribution yielding one by one elements from iterator `iter`,
in order.

# Examples
```jldoctest
julia> rand(Fill(Iterate(Iterators.countfrom(1)), 3), 2)
2-element Array{Array{Int64,1},1}:
 [1, 2, 3]
 [1, 2, 3]

julia> rand(Iterate(1:9), 10)
ERROR: ArgumentError: iterator exhausted
[...]
```

!!! note
    Iterate(iter) is semantically equivalent to the following definition:
    ```julia
    function Iterate(iter)
        local st
        Thunk() do
            it = @isdefined(st) ? iterate(iter, st) :
                                  iterate(iter)
            it === nothing && throw(ArgumentError("iterator exhausted"))
            st = it[2]
            Pure(it[1])
        end
    end
    ```
"""
struct Iterate{T,I} <: Distribution{T}
    iter::I

    Iterate(iter) = new{eltype(iter),typeof(iter)}(iter)
end

mutable struct SamplerIterate{T,I,E,S} <: SamplerReset{T}
    iter::I
    elem::Union{Some{E},Nothing} # Some only on first iteration
    state::S
end

function Sampler(::Type{RNG}, it::Iterate{T,I},
                 n::Repetition) where {RNG<:AbstractRNG,T,I}
    r = iterate(it.iter)
    if r === nothing
        SamplerIterate{T,Nothing,Nothing,Nothing}(nothing, nothing, nothing)
    else
        x, st = r
        SamplerIterate{T,I,typeof(x),typeof(st)}(it.iter,Some(x),st)
    end
end

function reset!(sp::SamplerIterate, _...)
    x, st = iterate(sp.iter) # we know it's not nothing
    sp.elem = Some(x)
    sp.state = st
    sp
end

reset!(sp::SamplerIterate{T,Nothing}, _...) where {T} = sp # nothing to do

function rand(rng::AbstractRNG, sp::SamplerIterate)
    if sp.elem !== nothing
        elem = sp.elem
        sp.elem = nothing
        return something(elem)
    end
    r = iterate(sp.iter, sp.state)
    r === nothing &&
        throw(ArgumentError("iterator exhausted"))
    sp.state = r[2]
    r[1]
end

rand(rng::AbstractRNG, sp::SamplerIterate{T,Nothing}) where {T} =
    throw(ArgumentError("iterator exhausted"))


## SubSeq ####################################################################

"""
    SubSeq(A::AbstractArray{T}, p::Real) where {T} :: Distribution{Vector{T}}

Create a distribution yielding vectors consisting of a random subsequence of
the given array `A`, where each element of `A` is included (in order) with
independent probability `p`. (Complexity is linear in `p*length(A)`, so this
is efficient even if `p` is small and `A` is large.) Technically,
this process is known as "Bernoulli sampling" of `A`.

# Examples
```julia-repl
julia> rand(SubSeq(1:8, 0.3))
3-element Array{Int64,1}:
 3
 4
 7

julia> rand!(Int[], SubSeq(1:8, 0.3), Val(1))
2-element Array{Int64,1}:
 4
 5
```

!!! note
    `SubSeq` is implemented as a thin wrapper over `Random.randsubseq!`, but
    is semantically equivalent to the following definition:
    ```julia
    SubSeq(A, p) = Lift(filter,
                        Bind(Fill(Bernoulli(p), length(A))) do selected
                            Pure(_ -> pop!(selected))
                        end,
                        Pure(A))
    ```
"""
struct SubSeq{T,A} <: Distribution{T}
    a::A
    p::Float64

    SubSeq(a::AbstractArray{T}, p::Real) where {T} =
        new{Vector{T},typeof(a)}(a, p)
end

rand!(rng::AbstractRNG, s::AbstractVector,
      sp::SamplerTrivial{<:SubSeq}, ::Val{1}) =
    randsubseq!(rng, s, sp[].a, sp[].p)

rand(rng::AbstractRNG, sp::SamplerTrivial{<:SubSeq{T}}) where {T} =
    rand!(rng, T(), sp, Val(1))


## SubIter ###################################################################

"""
    SubIter(iter, p::Real) :: Distribution{eltype(iter)}

Create a distribution yielding one by one some elements
from iterator `iter`, in order. Each element is included with
independent probability `p`. Note that
+ `SubIter(iter, 1)` is equivalent to `Iterate(iter)`
+ `rand(SubIter(array, p), n)` is somewhat equivalent to
  `rand(SubSeq(array, p))[1:n]`, assuming `n` is small enough so that
  not all the values of `array` are consumed.

# Examples
```julia-repl
julia> rand(Fill(SubIter(Iterators.countfrom(1), .2), 10), 2)
2-element Array{Array{Int64,1},1}:
 [4, 5, 6, 13, 17, 21, 23, 41, 42, 48]
 [2, 6, 12, 28, 49, 51, 58, 60, 65, 66]
```

!!! note
    `SubIter(iter, p)` is semantically equivalent (roughly) to:
    ```
    SubIter(iter, p) =
        Keep(x -> x !== nothing,
             Bind(Zip(Bernoulli(p), Iterate(iter))) do (succ, x)
                 Pure(succ ? x : nothing)
             end)
    ```
    This is assuming that `iter` doesn't contain `nothing`.
"""
struct SubIter{T,I} <: Distribution{T}
    iter::I
    p::Float64

    SubIter(iter, p::Real) = new{eltype(iter),typeof(iter)}(iter, p)
end

Sampler(::Type{RNG}, sub::SubIter, n::Repetition) where {RNG<:AbstractRNG} =
    SamplerTag{typeof(sub)}((sp = Sampler(RNG, Iterate(sub.iter), n),
                             bernouilli = Sampler(RNG, Bernoulli(sub.p), n)))

reset!(sp::SamplerTag{<:SubIter}, n...) = (reset!(sp.data.sp, n...); sp)

function rand(rng::AbstractRNG, sp::SamplerTag{<:SubIter{T}})::T where {T}
    it = sp.data.sp
    bernouilli = sp.data.bernouilli
    while true
        x = rand(rng, it)
        rand(rng, bernouilli) && return x
    end
end


## Shuffling #################################################################

### Fisher-Yates

struct FisherYates{T,A} <: Distribution{T}
    a::A

    FisherYates(a::AbstractArray{T}) where {T} =
        new{T,typeof(a)}(a)
end

struct SamplerFisherYates{T,A} <: SamplerReset{T}
    a::A
    inds::Vector{Int}
end

fy_inds(a) = Vector{Int}(undef, length(a) + 1)

Sampler(::Type{RNG}, fy::FisherYates{T,A}, ::Repetition) where {RNG<:AbstractRNG} where {T,A} =
    reset!(SamplerFisherYates{T,A}(fy.a, fy_inds(fy.a)))

fy_reset!(inds, n) = @inbounds inds[end] = -n # < 0 means not yet initialized

reset!(sp::SamplerFisherYates, n=length(sp.a)) = (fy_reset!(sp.inds, n); sp)

@noinline function fy_initialize!(rng, inds, k)
    k == 0 &&
        throw(ArgumentError("FisherYates: all elements have been consumed"))
    n = length(inds) - 1
    kn = n
    copyto!(inds, 1:n)
    m = n + k
    mask = nextpow(2, n) - 1
    while n != m
        (mask >> 1) == n && (mask >>= 1)
        i = 1 + rand(rng, Random.ltm52(n, mask))
        #^^^ faster equivalent to i = rand(rng, 1:n) (cf. Base.shuffle!)
        @inbounds inds[i], inds[n] = inds[n], inds[i]
        n -= 1
    end
    kn
end

function fy_rand(rng::AbstractRNG, inds, a)
    @inbounds begin
        k = inds[end] # contains the index in inds where the index in a is located
        if k <= 0
            k = fy_initialize!(rng, inds, k)
        end
        inds[end] = k - 1
        a[inds[k]]
    end
end

rand(rng::AbstractRNG, sp::SamplerFisherYates) = fy_rand(rng, sp.inds, sp.a)


### SelfAvoid

# cf. `self_avoid_sample!` in StatsBase.jl

struct SelfAvoid{T,A} <: Distribution{T}
    a::A

    SelfAvoid(a::AbstractArray{T}) where {T} =
        new{T,typeof(a)}(a)
end

struct SamplerSelfAvoid{T,A,S} <: SamplerReset{T}
    a::A
    seen::Set{Int}
    idx::S
end

sa_idx(::Type{RNG}, a) where {RNG} = Sampler(RNG, Base.OneTo(length(a)), Val(Inf))

function Sampler(RNG::Type{<:AbstractRNG}, sa::SelfAvoid{T,A}, ::Repetition) where {T,A}
    idx  = sa_idx(RNG, sa.a)
    SamplerSelfAvoid{T,A,typeof(idx)}(sa.a, Set{Int}(), idx)
end

sa_reset!(sp) = empty!(sp.seen)

reset!(sp::SamplerSelfAvoid, _=0) = (sa_reset!(sp); sp)

function sa_rand(rng::AbstractRNG, sp)
    seen = sp.seen
    idx = sp.idx
    while true
        i = rand(rng, idx)
        if !(i in seen)
            push!(seen, i)
            return @inbounds sp.a[i]
        end
    end
end

rand(rng::AbstractRNG, sp::SamplerSelfAvoid) = sa_rand(rng, sp)


## Shuffle ###################################################################

struct Shuffle{T,A} <: Distribution{T}
    a::A
    n::Int

    # univariate constructor
    Shuffle(a::AbstractArray{T}) where {T} = new{T,typeof(a)}(a)

    # multivariate constructor
    Shuffle(a::AbstractArray{T}, n::Integer) where {T} =
        new{Vector{T},typeof(a)}(a, n)

end

mutable struct SamplerShuffle{T,A,S<:Sampler} <: SamplerReset{T}
    a::A
    n::Int
    alg::Int
    idx::S
    seen::Set{Int}
    inds::Vector{Int}

    SamplerShuffle{TT}(a::AbstractArray{T}, n::Int, idx::S) where {TT,T,S} =
        new{TT,typeof(a),S}(a, n, 0, idx)
end

Sampler(::Type{RNG}, sh::Shuffle{T}, n::Repetition) where {RNG<:AbstractRNG,T} =
    SamplerShuffle{T}(sh.a, sh.n, sa_idx(RNG, sh.a))

isunivariate(::SamplerShuffle{T,A}) where {T,A} = T === eltype(A)

function _reset!(sp::SamplerShuffle, n=-1)::Int
    na = length(sp.a)
    if n < 0
        n = na
    end
    if n < 3
        sp.alg = typemin(Int)
    elseif na < n * 24 # TODO: do benchmarks
        if !isdefined(sp, :inds)
            sp.inds = fy_inds(sp.a)
        end
        fy_reset!(sp.inds, n)
        sp.alg = 3
    else
        if !isdefined(sp, :seen)
            sp.seen = Set{Int}()
        end
        sa_reset!(sp)
        sp.alg = 4
    end
end

function reset!(sp::SamplerShuffle, n...)
    isunivariate(sp) && _reset!(sp, n...)
    sp
end

function rand(rng::AbstractRNG, sp::SamplerShuffle{T}) where T
    isunivariate(sp) || return rand!(rng, T(undef, sp.n), sp, Val(1))
    alg = sp.alg
    if alg < 0
        local i
        while true
            i = rand(rng, sp.idx)
            -i != alg && break
        end
        sp.alg = -i
        return @inbounds sp.a[i]
    elseif alg == 3
        fy_rand(rng, sp.inds, sp.a)
    elseif alg == 4
        sa_rand(rng, sp)
    else
        rand(rng, reset!(sp))
    end
end

function rand!(rng::AbstractRNG, v::AbstractVector, sp::SamplerShuffle{T}, ::Val{1}) where T
    isunivariate(sp) &&
        throw(ArgumentError("can not create vector from univariate Shuffle"))
    n = sp.n
    resize!(v, n)
    if n < 3
        n < 1 && return v
        j = rand(rng, sp.idx)
        @inbounds v[1] = sp.a[j]
        n < 2 && return v
        local i
        while true
            i = rand(rng, sp.idx)
            i != j && break
        end
        @inbounds v[2] = sp.a[i]
        return v
    end
    _reset!(sp, sp.n)
    if sp.alg == 3
        for i in eachindex(v)
            @inbounds v[i] = fy_rand(rng, sp.inds, sp.a)
        end
    else # alg == 4
        for i in eachindex(v)
            @inbounds v[i] = sa_rand(rng, sp)
        end
    end
    return v
end
