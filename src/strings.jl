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

# document example:  rand(Str(Keep(!isspace, Keep(isprint, Keep(isascii, Char)))))
# or rand(Str(Keep(isprint, Char), 40), 10)

"""
    Str([chars], [len=8]) :: Distribution{String}

Create a distribution generating strings of length `len`, consisting of
characters from `chars`, which defaults to the set of upper- and lower-case
letters and the digits 0-9. A call to `ranstring(rng, x...)` is equivalent
to `rand(rng, Str(x...))`. `chars` can also be `Char`, to generate any
unicode characters, or any distribution generating characters.

# Examples
```julia-repl
julia> rand(Str())
"GHTYhgfT"

julia> rand(Str(Keep(isprint, Char)))
"쳳𩌸篍𝞁𩚦ḏ𐮆ꌲ"
```

See also [`AsciiStr`](@ref).
"""
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

"""
    AsciiStr([chars], [len=8]; check::Bool=true) :: Distribution{String}

Similar to `Str(chars, len)` except that only ASCII strings are produced.
This means that `chars` must specify only ASCII characters, which is checked
for if `check` is `true`. When `chars` is not specified, it defaults to the
set of printable ASCII characters (cf. `isprint`).

Unlike for `Str`, `chars` cannot be a distribution, and must be convertible
to `Vector{UInt8}`.

One benefit of `AsciiStr` over [`Str`](@ref) is efficiency, as the size of all
characters is known in advance. `AsciiStr(chars; check=false)` should have the
same performance `Str(collect(UInt8, chars)`.

# Example
```julia-repl
julia> rand(AsciiStr(20)) # only printable characters
"gw2.}pqF_r\\"/w- x?]2r"

julia> rand(AsciiStr(Char)) # any ASCII character
"tV\\x0eeb'\\b\\x04"
```
"""
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
