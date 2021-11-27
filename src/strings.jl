## Str #######################################################################

struct Str{C} <: Distribution{String}
    len::Int
    chars::C
end

let b = UInt8['0':'9';'A':'Z';'a':'z']
    global Str, Sampler, rand

    Str()                                 = Str(8, b)
    Str(chars)                            = Str(8, chars)
    Str(::Type{C}) where C                = Str(8, Uniform(C))
    Str(n::Integer)                       = Str(Int(n), b)
    Str(chars,      n::Integer)           = Str(Int(n), chars)
    Str(::Type{C},  n::Integer) where {C} = Str(Int(n), Uniform(C))
    Str(n::Integer, chars)                = Str(Int(n), chars)
    Str(n::Integer, ::Type{C}) where {C}  = Str(Int(n), Uniform(C))
    Str(n::Int, ::Type{C}) where {C}      = Str(n, Uniform(C)) # disambiguate

    Sampler(::Type{RNG}, str::Str, n::Repetition) where {RNG<:AbstractRNG} =
        SamplerTag{typeof(str)}(sampler(RNG, str.chars, Val(Inf)) => str.len)

    function rand(rng::AbstractRNG, sp::SamplerTag{<:Str})
        chars, len = sp.data
        T = gentype(chars)
        v = T === UInt8 ? Base.StringVector(len) : Vector{T}(undef, len)
        rand!(rng, v, chars)
        String(v)
    end
end
