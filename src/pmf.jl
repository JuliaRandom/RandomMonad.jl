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

for T in (Number,AbstractChar,String)
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

function pmf(A::Union{AbstractArray,Tuple}, x)
    x ∈ A || return 0.0
    c = count(isequal(x), A)
    c * 1.0 / length(A)
end

pmf(A::AbstractRange, x) = Float64(x ∈ A) / length(A)
