## Uniform

abstract type Uniform{T} <: Distribution{T} end


struct UniformType{T} <: Uniform{T} end

Uniform(::Type{T}) where {T} = UniformType{T}()

Base.getindex(::UniformType{T}) where {T} = T

struct UniformWrap{T,E} <: Uniform{E}
    val::T
end

Uniform(x::T) where {T} = UniformWrap{T,gentype(T)}(x)

Base.getindex(x::UniformWrap) = x.val


### sampling

Sampler(RNG::Type{<:AbstractRNG}, d::Union{UniformWrap,UniformType}, n::Repetition) =
    Sampler(RNG, d[], n)


## Bernoulli

struct Bernoulli{T<:Number} <: Distribution{T}
    p::Float64

    Bernoulli{T}(p::Real) where {T} = let pf = Float64(p)
        0.0 <= pf <= 1.0 ? new(pf) :
            throw(DomainError(p, "Bernoulli: parameter p must satisfy 0.0 <= p <= 1.0"))
    end
end

Bernoulli(p::Real=0.5) = Bernoulli(Int, p)
Bernoulli(::Type{T}, p::Real=0.5) where {T<:Number} = Bernoulli{T}(p)


### sampling

Sampler(RNG::Type{<:AbstractRNG}, b::Bernoulli, n::Repetition) =
    SamplerTag{typeof(b)}(b.p+1.0)

rand(rng::AbstractRNG, sp::SamplerTag{Bernoulli{T}}) where {T} =
    ifelse(rand(rng, CloseOpen12()) < sp.data, one(T), zero(T))


## Categorical

struct Categorical{T<:Number} <: Distribution{T}
    cdf::Vector{Float64}

    Categorical{T}(c::Categorical) where {T} = new{T}(c.cdf)

    function Categorical{T}(weigths) where T
        if !isa(weigths, AbstractArray)
            # necessary for accumulate
            # TODO: will not be necessary anymore in Julia 1.5
            weigths = collect(weigths)
        end
        weigths = vec(weigths)

        isempty(weigths) &&
            throw(ArgumentError("Categorical requires at least one category"))

        s = Float64(sum(weigths))
        cdf = accumulate(weigths; init=0.0) do x, y
            x + Float64(y) / s
        end
        @assert isapprox(cdf[end], 1.0) # really?
        cdf[end] = 1.0 # to be sure the algo terminates
        new{T}(cdf)
    end
end

Categorical(weigths) = Categorical{Int}(weigths)

Categorical(n::Number) =
    Categorical{typeof(n)}(Iterators.repeated(1.0 / Float64(n), Int(n)))

function Base.convert(::Type{Categorical{T}}, d) where {T}
    d isa Number && throw(ArgumentError(
        "can not convert a number to a Categorical distribution"))
    Categorical{T}(d)
end

Base.convert(::Type{Categorical{T}}, d::Categorical{T}) where {T} = d


### sampling

Sampler(RNG::Type{<:AbstractRNG}, c::Categorical, n::Repetition) =
    SamplerSimple(c, Sampler(RNG, CloseOpen(), n))

# unfortunately requires @inline to avoid allocating
@inline rand(rng::AbstractRNG, sp::SamplerSimple{Categorical{T}}) where {T} =
    let c = rand(rng, sp.data)
        T(findfirst(x -> x >= c, sp[].cdf))
    end

# NOTE:
# if length(cdf) is somewhere between 150 and 200, the following gets faster:
#   T(searchsortedfirst(sp[].cdf, rand(rng, sp.data)))


## Mixture Model

struct MixtureModel{T,CT<:Distribution} <: Distribution{T}
    components::Vector{CT}
    prior::Categorical{Int}

    function MixtureModel{T}(components::Vector{CT}, prior::Categorical) where {T,CT}
        length(components) != length(prior.cdf) && throw(ArgumentError(
            "the number of components does not match the length of prior"))

        new{T,CT}(components, prior)
    end
end

function MixtureModel(components, prior)
    prior = convert(Categorical{Int}, prior)
    T = reduce(typejoin, (gentype(x) for x in components))
    v = [d isa Distribution ? d : Uniform(d) for d in components]
    MixtureModel{T}(v, prior)
end


### sampling

Sampler(RNG::Type{<:AbstractRNG}, m::MixtureModel, n::Val{1}) =
    SamplerSimple(m, Sampler(RNG, m.prior, n))

@inline rand(rng::AbstractRNG, sp::SamplerSimple{<:MixtureModel}) =
    rand(rng, sp[].components[rand(rng, sp.data)])::gentype(sp)

# WARNING: expensive, for less than 100 number generations or so, use Val(1) Sampler
Sampler(RNG::Type{<:AbstractRNG}, m::MixtureModel{T}, n::Val{Inf}) where {T} =
    SamplerTag{typeof(m)}(
        (components = Sampler{<:T}[Sampler(RNG, c, n) for c in m.components],
         prior      = Sampler(RNG, m.prior, n)))

rand(rng::AbstractRNG, sp::SamplerTag{<:MixtureModel}) =
    rand(rng, sp.data.components[rand(rng, sp.data.prior)])::gentype(sp)


## Normal

abstract type Normal{T} <: Distribution{T} end

struct Normal01{T} <: Normal{T} end

struct Normalμσ{T} <: Normal{T}
    μ::T
    σ::T
end

const NormalTypes = Union{AbstractFloat,Complex{<:AbstractFloat}}

Normal(::Type{T}=Float64) where {T<:NormalTypes} = Normal01{T}()
Normal(μ::T, σ::T) where {T<:NormalTypes} = Normalμσ(μ, σ)
Normal(μ::T, σ::T) where {T<:Real} = Normalμσ(AbstractFloat(μ), AbstractFloat(σ))
Normal(μ, σ) = Normal(promote(μ, σ)...)


### sampling

rand(rng::AbstractRNG, ::SamplerTrivial{Normal01{T}}) where {T<:NormalTypes} =
    randn(rng, T)

Sampler(RNG::Type{<:AbstractRNG}, d::Normalμσ{T}, n::Repetition) where {T} =
    SamplerSimple(d, Sampler(RNG, Normal(T), n))

rand(rng::AbstractRNG, sp::SamplerSimple{Normalμσ{T},<:Sampler}) where {T} =
    sp[].μ + sp[].σ  * rand(rng, sp.data)


## Exponential

abstract type Exponential{T} <: Distribution{T} end

struct Exponential1{T} <: Exponential{T} end

struct Exponentialθ{T} <: Exponential{T}
    θ::T
end

Exponential(::Type{T}=Float64) where {T<:AbstractFloat} = Exponential1{T}()
Exponential(θ::T) where {T<:AbstractFloat} = Exponentialθ(θ)
Exponential(θ::Real) = Exponentialθ(AbstractFloat(θ))


### sampling

rand(rng::AbstractRNG, ::SamplerTrivial{Exponential1{T}}) where {T<:AbstractFloat} =
    randexp(rng, T)

Sampler(RNG::Type{<:AbstractRNG}, d::Exponentialθ{T}, n::Repetition) where {T} =
    SamplerSimple(d, Sampler(RNG, Exponential(T), n))

rand(rng::AbstractRNG, sp::SamplerSimple{Exponentialθ{T},<:Sampler}) where {T} =
    sp[].θ * rand(rng, sp.data)
