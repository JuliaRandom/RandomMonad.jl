## discretize!

"""
    discretize!(a::AbstractArray{<:Real}, n::Integer=30)

Sort `a`, divide its values into (roughly) `n` groups corresponding to
equal-length intervals, and replace each value in a group by the `mean`
of this group. This can be useful to transform samples from a continuous
distribution in order to plot an histogram.

# Examples
```julia-repl
julia> pmf(discretize!(randn(1000), 10))
pmf for [-3.2977741401356235, -2.804784333001702, -2.804784333001702, -2.804784333001702, -2.0873488466856744, -2.0873488466856744, -2.0873488466856744, -2.0873488466856744, -2.0873488466856744, -2.0873488466856744  …  2.0792314833569865, 2.0792314833569865, 2.0792314833569865, 2.0792314833569865, 2.0792314833569865, 2.9112746357518313, 2.9112746357518313, 2.9112746357518313, 2.9112746357518313, 2.9112746357518313] with support of length 10:
  -3.29777  => 0.001
  -2.80478  => 0.003
  -2.08735  => 0.026
  -1.36503  => 0.104
  -0.679956 => 0.237
  0.0211377 => 0.273
  0.748744  => 0.232
  1.46951   => 0.101
  2.07923   => 0.018
  2.91127   => 0.005
```
"""
function discretize!(a::AbstractArray{<:Real}, n::Integer=30)
    require_one_based_indexing(a)
    length(a) < 2n && return a
    n = min(n, length(a) ÷ 2)
    sort!(a)
    w = (a[end] - a[1]) / n
    i = 1
    for x in LinRange(a[1], a[end], n)
        j = searchsortedlast(a, x)
        y = mean(view(a, i:j))
        a[i:j] .= y
        i = j+1
    end
    a
end


## DictWrap

struct DictWrap{W, K, V} <: AbstractDict{K, V}
    wrap::W

    DictWrap(wrap) = new{typeof(wrap), keytype(wrap), valtype(wrap)}(wrap)
end

Base.getindex(dw::DictWrap, x) = dw.wrap[x]
Base.keys(dw::DictWrap) = keys(dw.wrap)
Base.values(dw::DictWrap) = values(dw.wrap)
Base.length(dw::DictWrap) = length(dw.wrap)
Base.summary(io::IO, dw::DictWrap) = Base.summary(io, dw.wrap)
Base.typeinfo_prefix(io::IO, dw::DictWrap) =
    sprint(Base.show_type_name, typeof(dw.wrap).name), false

Base.iterate(dw::DictWrap, st...) = Base.iterate(dw.wrap, st...)

Base.keytype(d::Distribution) = keytype(typeof(d))
Base.valtype(d::Distribution) = valtype(typeof(d))
