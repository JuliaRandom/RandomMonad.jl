## AbstractStr ###############################################################

abstract type AbstractStr <: Distribution{String} end

function Sampler(::Type{RNG}, str::AbstractStr, n::Repetition
                 ) where {RNG<:AbstractRNG}
    SamplerTag{typeof(str)}(sampler(RNG, str.chars, Val(Inf)) => str.len)
end

function rand(rng::AbstractRNG, sp::SamplerTag{<:AbstractStr})
    chars, len = sp.data
    T = gentype(chars)
    v = T === UInt8 ? Base.StringVector(len) : Vector{T}(undef, len)
    rand!(rng, v, chars)
    String(v)
end


## Str #######################################################################

struct Str{C} <: AbstractStr
    len::Int
    chars::C
end

let b = UInt8['0':'9';'A':'Z';'a':'z']
    global Str

    Str()                                 = Str(8, b)
    Str(chars)                            = Str(8, chars)
    Str(::Type{C}) where C                = Str(8, Uniform(C))
    Str(n::Integer)                       = Str(Int(n), b)
    Str(chars,      n::Integer)           = Str(Int(n), chars)
    Str(::Type{C},  n::Integer) where {C} = Str(Int(n), Uniform(C))
    Str(n::Integer, chars)                = Str(Int(n), chars)
    Str(n::Integer, ::Type{C}) where {C}  = Str(Int(n), Uniform(C))
    Str(n::Int, ::Type{C}) where {C}      = Str(n, Uniform(C)) # disambiguate
end


## AsciiStr ##################################################################

struct AsciiStr <: AbstractStr
    len::Int
    chars::Vector{UInt8}

    function AsciiStr(len::Integer, chars; check::Bool=true)
        chars8 = chars isa Vector{UInt8} ? chars : collect(UInt8, chars)
        if check
            ok = chars isa String ? isascii(chars) : all(_isascii, chars8)
            ok || throw(ArgumentError("`chars` must represent ascii characters"))
        end
        new(len, chars8)
    end
end

_isascii(c::UInt8) = c <= 0x80

let ascii = collect(0x00:0x80),
    printable = filter(x -> isprint(Char(x)), 0x00:0x80)
    global AsciiStr

    AsciiStr(len::Integer=8; check::Bool=false) =
        AsciiStr(len, printable; check=false)
    AsciiStr(chars, n::Integer=8; check::Bool=true) =
        AsciiStr(n, chars; check=check)
    AsciiStr(n::Integer, ::Type{Char}; check::Bool=false) =
        AsciiStr(n, ascii; check=false)
    AsciiStr(::Type{Char}, n::Integer=8; check::Bool=false) =
        AsciiStr(n, ascii; check=false)
end
