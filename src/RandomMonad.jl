module RandomMonad

export Distribution

export Bernoulli, Bind, Binomial, Categorical, CloseClose, CloseOpen, Const,
       counts, Counts, Exponential, Fill, Filter, FisherYates, Lift,
       MixtureModel, Multinomial, Normal, OpenClose, OpenOpen, Poisson, Reduce,
       SelfAvoid, Shuffle, Uniform, Unique, Zip

using Random: AbstractRNG, gentype, randexp, randn, Random, Repetition,
      SamplerSimple, SamplerTag, SamplerTrivial

import Random: rand, rand!, Sampler


"""
    Distribution{T}

An instance of a subtype of `Distribution{T}` is an object able
to produce random values of type `T` via `rand`-related functions.
"""
abstract type Distribution{T} end

"""
    Base.eltype(::Type{<:Distribution{T}}) where {T}

Return the type parameter `T` of a distribution, i.e. the type of
values drawn from it.
"""
Base.eltype(::Type{<:Distribution{T}}) where {T} = T

"""
    reset!(sp::Random.Sampler, [n::Integer]) -> sp

Prepare the sampler state for the generation of `n` new random objects, or of
an arbitrary number when `n` is not specified, and return `sp`. This is useful
for samplers which have "memory", where an internal state has to be
re-initialized. The default implementation has no side effects.

When the sampler type for a given distribution depends on the `Repetition`
argument `rep` passed to its constructor, `reset!` generally makes sense only
for the case `rep == Val(Inf)`.

This function should be called in each function generating a collection of
objects generated from a given sampler `sp`. As a notable exception,
`rand!(::AbstractRNG, ::Array, ::Sampler)` does not follow this rule, as it is
defined in `Random`. Therefore, samplers having a non-trivial `reset!` should
not assume that this function will be called (one easy way to handle that is to
call `reset!` at construction time).
"""
reset!(sp::Sampler, _ = 0) = sp


## extended rand!

"""
    rand!([rng::AbstractRNG], A::AbstractArray, X::Distribution, ::Val{1})

Populate the array `A` with random values, according to the distribution specified
by `X`, whose `eltype` must be compatible with the type of `A`.
`A` will be resized as needed when possible.

!!! note
    The standard `rand!(A, X)` syntax is used to populate `A` according to
    distribution `X` which describes how to generate individual elements. On
    the other hand, in `rand!(A, X, Val(1))`, `X` describes how to generate a
    full array.

# Examples
```julia
julia> rand!(zeros(Int, 3), Fill(1:3, 4), Val(1))
4-element Array{Int64,1}:
 2
 3
 2
 1

julia> rand!(zeros(Int, 2, 2), Fill(1:3, 4, 1), Val(1))
ERROR: ArgumentError: can not resize destination array
Stacktrace:
[...]
```
"""
rand!(rng::AbstractRNG, A::AbstractArray, X::Distribution, ::Val{1}) =
    rand!(rng, A, Sampler(rng, X), Val(1))

rand!(A::AbstractArray, X::Distribution, ::Val{1}) = rand!(Random.GLOBAL_RNG, A, X, Val(1))


## includes

include("distributions.jl")
include("floatintervals.jl")
include("containers.jl")
include("adapters.jl")


end # module
