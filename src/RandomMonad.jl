module RandomMonad

export Distribution

export Bernoulli, Binomial, Categorical, CloseOpen, OpenClose, OpenOpen,
       CloseClose, Exponential, Filter, Map, MixtureModel, Normal, Uniform

using Random: AbstractRNG, gentype, randexp, randn, Random, Repetition,
      SamplerSimple, SamplerTag, SamplerTrivial

import Random: rand, Sampler


"""
    Distribution{T}

An instance of a subtype of `Distribution{T}` is an object able
of produce random values of type `T` via `rand`-related functions.
"""
abstract type Distribution{T} end


"""
    Base.eltype(::Type{<:Distribution{T}}) where {T}

Return the type parameter `T` of a distribution, i.e. the type of
values drawn from it.
"""
Base.eltype(::Type{<:Distribution{T}}) where {T} = T


include("distributions.jl")
include("floatintervals.jl")
include("adapters.jl")


end # module
