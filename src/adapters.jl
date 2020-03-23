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
