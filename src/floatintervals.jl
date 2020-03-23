abstract type FloatInterval{T<:AbstractFloat} <: Uniform{T} end

abstract type CloseOpen{ T<:AbstractFloat} <: FloatInterval{T} end
abstract type OpenClose{ T<:AbstractFloat} <: FloatInterval{T} end
abstract type CloseClose{T<:AbstractFloat} <: FloatInterval{T} end
abstract type OpenOpen{  T<:AbstractFloat} <: FloatInterval{T} end

struct CloseOpen12{T<:AbstractFloat} <: CloseOpen{T} end # interval [1,2)

struct CloseOpen01{ T<:AbstractFloat} <: CloseOpen{T}  end # interval [0,1)
struct OpenClose01{ T<:AbstractFloat} <: OpenClose{T}  end # interval (0,1]
struct CloseClose01{T<:AbstractFloat} <: CloseClose{T} end # interval [0,1]
struct OpenOpen01{  T<:AbstractFloat} <: OpenOpen{T}   end # interval (0,1)

struct CloseOpenAB{T<:AbstractFloat} <: CloseOpen{T} # interval [a,b)
    a::T
    b::T

    CloseOpenAB{T}(a::T, b::T) where {T} = (check_interval(a, b); new{T}(a, b))
end

struct OpenCloseAB{T<:AbstractFloat} <: OpenClose{T} # interval (a,b]
    a::T
    b::T

    OpenCloseAB{T}(a::T, b::T) where {T} = (check_interval(a, b); new{T}(a, b))
end

struct CloseCloseAB{T<:AbstractFloat} <: CloseClose{T} # interval [a,b]
    a::T
    b::T

    CloseCloseAB{T}(a::T, b::T) where {T} = (check_interval(a, b); new{T}(a, b))
end

struct OpenOpenAB{T<:AbstractFloat} <: OpenOpen{T} # interval (a,b)
    a::T
    b::T

    OpenOpenAB{T}(a::T, b::T) where {T} = (check_interval(a, b); new{T}(a, b))
end

check_interval(a, b) = a >= b && throw(ArgumentError("invalid interval specification"))

const FloatInterval_64 = FloatInterval{Float64}
const CloseOpen01_64   = CloseOpen01{Float64}
const CloseOpen12_64   = CloseOpen12{Float64}

CloseOpen01(::Type{T}=Float64) where {T<:AbstractFloat} = CloseOpen01{T}()
CloseOpen12(::Type{T}=Float64) where {T<:AbstractFloat} = CloseOpen12{T}()

CloseOpen(::Type{T}=Float64) where {T<:AbstractFloat} = CloseOpen01{T}()
CloseOpen(a::T, b::T) where {T<:AbstractFloat} = CloseOpenAB{T}(a, b)

OpenClose(::Type{T}=Float64) where {T<:AbstractFloat} = OpenClose01{T}()
OpenClose(a::T, b::T) where {T<:AbstractFloat} = OpenCloseAB{T}(a, b)

CloseClose(::Type{T}=Float64) where {T<:AbstractFloat} = CloseClose01{T}()
CloseClose(a::T, b::T) where {T<:AbstractFloat} = CloseCloseAB{T}(a, b)

OpenOpen(::Type{T}=Float64) where {T<:AbstractFloat} = OpenOpen01{T}()
OpenOpen(a::T, b::T) where {T<:AbstractFloat} = OpenOpenAB{T}(a, b)

# convenience functions

CloseOpen(a, b) = CloseOpen(promote(a, b)...)
CloseOpen(a::T, b::T) where {T} = CloseOpen(AbstractFloat(a), AbstractFloat(b))

OpenClose(a, b) = OpenClose(promote(a, b)...)
OpenClose(a::T, b::T) where {T} = OpenClose(AbstractFloat(a), AbstractFloat(b))

CloseClose(a, b) = CloseClose(promote(a, b)...)
CloseClose(a::T, b::T) where {T} = CloseClose(AbstractFloat(a), AbstractFloat(b))

OpenOpen(a, b) = OpenOpen(promote(a, b)...)
OpenOpen(a::T, b::T) where {T} = OpenOpen(AbstractFloat(a), AbstractFloat(b))


## sampling

### fall-back on Random definitions

for CO in (:CloseOpen01, :CloseOpen12)
    @eval begin
        Sampler(RNG::Type{<:AbstractRNG}, ::$CO{T}, n::Repetition) where {T} =
            Sampler(RNG, Random.$CO{T}(), n)

        Sampler(::Type{<:AbstractRNG}, ::$CO{BigFloat}, ::Repetition) =
            Random.SamplerBigFloat{Random.$CO{BigFloat}}(precision(BigFloat))
    end
end


### new intervals 01

# TODO: optimize for BigFloat

for CO = (:OpenClose01, :OpenOpen01, :CloseClose01)
    @eval Sampler(RNG::Type{<:AbstractRNG}, I::$CO{T}, n::Repetition) where {T} =
              SamplerSimple(I, CloseOpen01(T))
end

rand(r::AbstractRNG, sp::SamplerSimple{OpenClose01{T}}) where {T} =
    one(T) - rand(r, sp.data)

rand(r::AbstractRNG, sp::SamplerSimple{OpenOpen01{T}}) where {T} =
    while true
        x = rand(r, sp.data)
        x != zero(T) && return x
    end

# optimizations (TODO: optimize for BigFloat too)

rand(r::AbstractRNG, sp::SamplerSimple{OpenOpen01{Float64}}) =
    reinterpret(Float64, reinterpret(UInt64, rand(r, sp.data)) | 0x0000000000000001)

rand(r::AbstractRNG, sp::SamplerSimple{OpenOpen01{Float32}}) =
    reinterpret(Float32, reinterpret(UInt32, rand(r, sp.data)) | 0x00000001)

rand(r::AbstractRNG, sp::SamplerSimple{OpenOpen01{Float16}}) =
    reinterpret(Float16, reinterpret(UInt16, rand(r, sp.data)) | 0x0001)

# prevfloat(T(2)) - 1 for IEEEFloat
upper01(::Type{Float64}) = 0.9999999999999998
upper01(::Type{Float32}) = 0.9999999f0
upper01(::Type{Float16}) = Float16(0.999)
upper01(::Type{BigFloat}) = prevfloat(one(BigFloat))

rand(r::AbstractRNG, sp::SamplerSimple{CloseClose01{T}}) where {T} =
    rand(r, sp.data) / upper01(T)


### CloseOpenAB

for (CO, CO01) = (CloseOpenAB => CloseOpen01,
                  OpenCloseAB => OpenClose01,
                  CloseCloseAB => CloseClose01,
                  OpenOpenAB => OpenOpen01)

    @eval Sampler(RNG::Type{<:AbstractRNG}, d::$CO{T}, n::Repetition) where {T} =
        SamplerTag{$CO{T}}((a=d.a, d=d.b - d.a, sp=Sampler(RNG, $CO01{T}(), n)))

    @eval rand(rng::AbstractRNG, sp::SamplerTag{$CO{T}}) where {T} =
        sp.data.a + sp.data.d  * rand(rng, sp.data.sp)
end
