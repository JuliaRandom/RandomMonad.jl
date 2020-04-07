## Pure ######################################################################

"""
    Pure(x) :: Distribution{typeof{x}}

Wrap `x` as a distribution always yielding `x`.
`Pure(x)` is equivalent to the implicit distributions
`[x]` and `(x,)`.

# Examples
```jldoctest
julia> rand(Pure(1), 3)
3-element Array{Int64,1}:
 1
 1
 1

julia> rand([1], 3)
3-element Array{Int64,1}:
 1
 1
 1
```

!!! note
    The name `Pure` comes from the similar function of the "applicative"
    Haskell typeclass.
"""
struct Pure{T} <: Distribution{T}
    x::T
end

rand(::AbstractRNG, sp::SamplerTrivial{<:Pure}) = sp[].x


## algebra ###################################################################

struct Op2{T,F,A,B} <: Distribution{T}
    f::F
    a::A
    b::B

    Op2{T}(f::F, a::A, b::B) where {T,F,A,B} = new{T,F,A,B}(f, a, b)
end

Op2(f::F, a::A, b::B) where {F,A,B} =
    Op2{typeof(f(one(gentype(a)), one(gentype(b))))}(f, a, b)

Sampler(RNG::Type{<:AbstractRNG}, x::Op2, n::Repetition) =
    SamplerTag{typeof(x)}((f = x.f,
                           a = Sampler(RNG, x.a, n),
                           b = Sampler(RNG, x.b, n)))

function reset!(sp::SamplerTag{<:Op2}, n...)
    reset!(sp.data.a, n...)
    reset!(sp.data.b, n...)
    sp
end

rand(rng::AbstractRNG, sp::SamplerTag{<:Op2{T}}) where {T} =
    sp.data.f(rand(rng, sp.data.a), rand(rng, sp.data.b))::T


### instances

for op = (:+, :-, :*, :/, :^)
    @eval begin
        (Base.$op)(a::Distribution, b::Distribution) = Op2($op, a,       b)
        (Base.$op)(a,               b::Distribution) = Op2($op, Pure(a), b)
        (Base.$op)(a::Distribution, b              ) = Op2($op, a,       Pure(b))
    end
end


### getindex

Op2(::typeof(getindex), a::A, b::B) where {A,B} =
    Op2{eltype(gentype(a))}(getindex, a, b)

Op2(::typeof(getindex), a::A, b::B) where {A<:Distribution{<:AbstractDict},B} =
    Op2{valtype(gentype(a))}(getindex, a, b)

"""
    getindex(X::Distribution, Y::Distribution) :: Distribution

Return a distribution yielding `x[y]` where `x <- X` and `y <- Y`.

# Examples
```julia
julia> rand(Pure('a':'z')[Uniform(1:3)])
'b': ASCII/Unicode U+0062 (category Ll: Letter, lowercase)
```
"""
Base.getindex(a::Distribution, b::Distribution) = Op2(getindex, a, b)


## Bind ######################################################################

"""
    Bind(f, X)

Generate a random value `x` from distribution `X`, and then
generate a random value from distribution specified by `f(x)`.

!!! note
    This is roughly equivalent to the `>>=` bind monadic operator
    in Haskell, but with the order of arguments reversed.

# Examples

The following expression generates 4 arrays of `Bool` (`Fill(Bool, n)`),
whose length `n` is chosen at random in `1:5`:
```julia
julia> rand(Bind(n -> Fill(Bool, n), 1:5), 4)
4-element Array{Array{Bool,1},1}:
 [1, 0, 1]
 [1, 1]
 [1, 0, 0, 1, 0]
 [1]
```
If multiple values must be passed to the function, `Zip` can be used
to combine distributions. For example, the following `Bind` construction
generates arrays of length yielded by `Shuffle(1:4)` (i.e. each array has
a distinct length in `1:4`), and whose elements are generated by a
distribution selected randomly in `[Bool, Normal(), 1:9]`:
```julia
julia> b = Bind(Zip([Bool, Normal(), 1:9], Shuffle(1:4))) do (t, n)
               Fill(t, n)
           end;

julia> rand(b, 4)
4-element Array{Any,1}:
 [-0.2900266833838387]
 Bool[0, 0, 1]
 Bool[0, 1, 1, 0]
 [6, 1]
```

!!! note
    The distribution specified by `f(x)` is not known until
    `rand` is called, which means that the "staged" approach
    enabled by the `Sampler` machinery cannot be fully
    exploited. So while `Bind` is a powerful general tool,
    specialized implementations of certain constructions might
    achieve better performance.

    Moreover, only `X` can have a stateful sampler (non-trivial
    `reset!` method), the distribution returned by `f` can not:
    indeed, a new sampler is created every time.
    For example, `Bind(n -> Iterate(1:n), 1:9)` always yields `1`.
"""
struct Bind{F,X,T} <: Distribution{T}
    f::F
    x::X

    function Bind{T}(f::F, x) where {T,F}
        x = wrap(x)
        new{F,typeof(x),T}(f, x)
    end
end

function Bind(f::F, x) where F
    rt = Base.return_types(f, (gentype(x),))
    T = length(rt) > 1 ? Any : rt[1]
    Bind{gentype(T)}(f, x)
end

Sampler(::Type{RNG}, b::Bind, n::Repetition) where {RNG<:AbstractRNG} =
    SamplerTag{typeof(b)}((x = sampler(RNG, b.x, n),
                           f = b.f))

reset!(sp::SamplerTag{<:Bind}, n...) = (reset!(sp.data.x); sp)

function rand(rng::AbstractRNG, sp::SamplerTag{<:Bind})
    x = rand(rng, sp.data.x)
    rand(rng, sp.data.f(x))
end


## Join ######################################################################

"""
    Join(x) :: Distribution

Given a distribution `x` yielding another distribution yielding `y`,
`Join(x)` is a distribution yielding `y`. This is equivalent to
`Bind(identity, x)`. When `x::Distribution{Distribution{T}}`, then
`Join(x)::Distribution{T}`; in other words, `Join` somehow "flattens"
two layers of "`Distribution` wrapping" into one.

# Examples
```julia-repl
julia> rand(Join(Pure(Normal())))
1.3938116112906205

julia> rand(Join([Normal(), Normal(10.0, 1.0)]), 5)
5-element Array{Float64,1}:
  0.8017321165720984
  9.987814723449322
 10.886780925603274
  0.17351815418302222
 11.606547172156468
```

!!! note
    * `Join` can be combined together with `Categorical` to implement
      "mixture models", e.g.
    ```julia
    MixtureModel(components, prior) = Join(Pure(components)[Categorical(prior)])
    ```

    * `Join` can be easily expressed in terms of `Bind`, but is no less
    fundamental, given `Lift`. Indeed, `Bind` could be defined as follows:
    ```julia
    Bind(f, x) = Join(Lift(f, x))
    ```
"""
struct Join{X,T} <: Distribution{T}
    x::X

    function Join(x::X) where X
        U = gentype(x)
        new{X,gentype(U)}(x)
    end
end

Sampler(::Type{RNG}, j::Join, n::Repetition) where {RNG<:AbstractRNG} =
    SamplerTag{typeof(j)}(Sampler(RNG, j.x, n))

rand(rng::AbstractRNG, sp::SamplerTag{<:Join}) =
    rand(rng, rand(rng, sp.data))



## Thunk #####################################################################

"""
    Thunk(f) :: Distribution

Create a distribution whose each generated value is the result of calling
`d = f()` and then returning a random value drawn from distribution `d`.
This is a particular case of [`Bind`](@ref), as `Thunk(f)` is equivalent
to `Bind(_ -> f(), Pure(nothing))`.

# Examples
```julia-repl
julia> rand(let i::Int = 0
                Thunk() do
                   i += 1
                   Uniform(1:i)
                end
            end, 5)
5-element Array{Int,1}:
 1
 1
 3
 4
 2
```
"""
struct Thunk{F,T} <: Distribution{T}
    f::F

    Thunk{T}(f::F) where {T,F} = new{F,T}(f)
end

function Thunk(f::F) where F
    rt = Base.return_types(f, ())
    T = length(rt) > 1 ? Any : rt[1]
    Thunk{gentype(T)}(f)
end

rand(rng::AbstractRNG, sp::SamplerTrivial{<:Thunk}) =
    rand(rng, sp[].f())


## Lift ######################################################################

"""
    Lift(f, Xs...) :: Distribution

Create a "lifted" version of `f` to the `Distribution` domain, which, given
yielded values `xs...` from `Xs...`, yields `f(xs...)`.

# Examples
```julia-repl
julia> rand(Lift(+, 0:10:20, Normal()), 5)
5-element Array{Float64,1}:
 -1.8066739854479257
 20.285455058446814
  8.674837595869976
  0.7620819803820099
  9.891731903551195
```

!!! note
    When distributions are seen as iterators, `Lift(f, Xs...)` is the iterator
    resulting from `map`ping `f` onto `Xs...`.
"""
struct Lift{T,F,D} <: Distribution{T}
    f::F
    d::D
end

Lift{T}(f::F, d...) where {T,F} = Lift{T,F,typeof(d)}(f, d)

function Lift(f::F, d...) where {F}
    rt = Base.return_types(f, map(gentype, d))
    T = length(rt) > 1 ? Any : rt[1]
    Lift{T}(f, d...)
end


### sampling

# Repetition -> Val(1)
rand(rng::AbstractRNG, sp::SamplerTrivial{<:Lift{T}}) where {T} =
    convert(T, sp[].f((rand(rng, d) for d in sp[].d)...))

Sampler(RNG::Type{<:AbstractRNG}, m::Lift, n::Val{Inf}) =
    SamplerTag{typeof(m)}((f = m.f,
                           d = map(x -> Sampler(RNG, x, n), m.d)))

reset!(sp::SamplerTag{<:Lift}, n...) =
    (foreach(s -> reset!(s, n...), sp.data.d); sp)

rand(rng::AbstractRNG, sp::SamplerTag{<:Lift{T}}) where {T} =
    convert(T, sp.data.f((rand(rng, d) for d in sp.data.d)...))


## Keep ######################################################################

"""
    Keep(f, X) :: Distribution

Create a distribution yielding a subset of the values yielded by `X`, keeping
only those for which `f` is `true`.

# Examples
```julia-repl
julia> rand(Keep(iseven, 1:9), 5)
5-element Array{Int64,1}:
 2
 8
 2
 8
 4
```

!!! note
    `Keep` is semantically equivalent to the following construction:
    ```julia
    Keep(f, X) = Bind(X) do x
                     f(x) ? Pure(x) :
                            Keep(f, X)
                 end
    ```
"""
struct Keep{T,F,D<:Distribution{T}} <: Distribution{T}
    f::F
    d::D
end

Keep(f::F, d) where {F} = Keep(f, Uniform(d))


### sampling

Sampler(::Type{RNG}, d::Keep, n::Repetition) where {RNG<:AbstractRNG} =
    SamplerTag{typeof(d)}((f = d.f,
                           d = Sampler(RNG, d.d, n)))

reset!(sp::SamplerTag{<:Keep}, n=0) = (reset!(sp.data.d, n); sp)

rand(rng::AbstractRNG, sp::SamplerTag{<:Keep}) =
    while true
        x = rand(rng, sp.data.d)
        sp.data.f(x) && return x
    end


## Map #######################################################################

"""
    Map(f, D...)

Given distributions `D...` yielding collections `d...`, create a distribution
yielding `map(f, d...)`.

!!! note
    `Map(f, D...)` is semantically equivalent to `Lift(map, Pure(f), D...)`.

# Examples
```julia
julia> rand(Map(+, Fill(0:10:20, 4), Fill(Normal(), 4)))
4-element Array{Float64,1}:
 20.51571632364027
 10.458305495441273
  0.24391036203770697
 18.700973042033308
```
"""
struct Map{T,F,D} <: Distribution{T}
    f::F
    d::D
end

Map{T}(f::F, d...) where {T,F} = Map{T,F,typeof(d)}(f, d)

function Map(f::F, d...) where {F}
    rt = Base.return_types(map, (F, map(gentype, d)...))
    T = length(rt) > 1 ? Any : rt[1]
    Map{T}(f, d...)
end


### sampling

rand(rng::AbstractRNG, sp::SamplerTrivial{<:Map{T}}) where {T} =
    convert(T, map(sp[].f, (rand(rng, d) for d in sp[].d)...))

Sampler(RNG::Type{<:AbstractRNG}, m::Map, n::Val{Inf}) =
    SamplerTag{typeof(m)}((f = m.f,
                           d = map(x -> Sampler(RNG, x, n), m.d)))

reset!(sp::SamplerTag{<:Map}, n...) =
    (foreach(s -> reset!(s, n...), sp.data.d); sp)

rand(rng::AbstractRNG, sp::SamplerTag{<:Map{T}}) where {T} =
    convert(T, map(sp.data.f, (rand(rng, d) for d in sp.data.d)...))


## Filter ####################################################################

"""
    Filter(   f, X) :: Distribution{eltype(X)}
    Filter{T}(f, X) :: Distribution{T}

Given distribution `X` yielding collection `x`, create a distribution
yielding `filter(f, x)`.
In the first form, `Filter(f, X)`, the collection `filter(f, x)` is
assumed to be of the same type as `x`. If this is not the case and
it has type `T` instead, use the second form `Filter{T}(f, X)`.

!!! note
    `Filter(f, X)` is semantically equivalent to `Lift(filter, Pure(f), X)`,
    which can be used as an alternative to `Filter{T}(f, X)` when `T` is not
    known.

# Examples
```julia-repl
julia>  rand(Filter(x -> x != 0, Fill(-1:1, 6)))
4-element Array{Int64,1}:
 -1
 -1
  1
 -1

julia> rand(Filter(x -> x != 0, Zip(-1:1, -1:1)))
ERROR: MethodError: Cannot `convert` an object of type
  Tuple{} to an object of type
  Tuple{Int64}
[...]

julia> rand(Filter{Tuple}(x -> x != 0, Zip(-1:1, -1:1)), 3)
3-element Array{Tuple,1}:
 (-1,)
 ()
 (1, -1)

julia> rand(Lift(filter, Pure(x -> x != 0), Zip(-1:1, -1:1)), 3)
3-element Array{Tuple{Vararg{Int64,N} where N},1}:
 (-1, 1)
 ()
 (1, -1)
```
"""
struct Filter{T,F,D} <: Distribution{T}
    f::F
    d::D
end

Filter{T}(f::F, d) where {T,F} = Filter{T,F,typeof(d)}(f, d)

Filter(f::F, d) where {F} = Filter{eltype(d)}(f, d)


### sampling

Sampler(RNG::Type{<:AbstractRNG}, r::Filter, n::Repetition) =
    SamplerTag{typeof(r)}((f = r.f,
                           d = Sampler(RNG, r.d, n)))

reset!(sp::SamplerTag{<:Filter}, n...) = (reset!(sp.data.d, n...); sp)

rand(rng::AbstractRNG, sp::SamplerTag{<:Filter{T}}) where {T} =
    convert(T, filter(sp.data.f, rand(rng, sp.data.d)))


## Reduce ####################################################################

struct Reduce{T,F,D} <: Distribution{T}
    f::F
    d::D
end

Reduce{T}(f::F, d) where {T,F} = Reduce{T,F,typeof(d)}(f, d)

# we only support reduce for f(::X, ::X) -> X
# use Lift + Base.reduce for more complicated cases
Reduce(f::F, d) where {F} = Reduce{eltype(gentype(d))}(f, d)


### sampling

rand(rng::AbstractRNG, sp::SamplerTrivial{<:Reduce{T}}) where {T} =
    convert(T, reduce(sp[].f, rand(rng, sp[].d)))

Sampler(RNG::Type{<:AbstractRNG}, r::Reduce, n::Val{Inf}) =
    SamplerTag{typeof(r)}((f = r.f,
                           d = Sampler(RNG, r.d, n)))

reset!(sp::SamplerTag{<:Reduce}, n=0) = (reset!(sp.data.d, n); sp)

rand(rng::AbstractRNG, sp::SamplerTag{<:Reduce{T}}) where {T} =
    convert(T, reduce(sp.data.f, rand(rng, sp.data.d)))


## Counts

"""
    Counts(x) :: Distribution{<:Dict}

Create a distribution yielding a dictionary whose keys are the elements
of the collection yielded by distribution `x`, and whose values are the
number of times each element appeared in the collection.

# Examples
```julia
julia> rand(Counts(Fill(Categorical([1/6, 2/6, 3/6]), 600)))
Dict{Int64,Int64} with 3 entries:
  2 => 193
  3 => 306
  1 => 101
```
"""
struct Counts{T,X} <: Distribution{Dict{T,Int}}
    x::X

    Counts(x::X) where {X} = new{eltype(gentype(x)),X}(x)
end

Sampler(::Type{RNG}, c::Counts, n::Repetition) where {RNG<:AbstractRNG} =
    SamplerTag{typeof(c)}(Sampler(RNG, c.x, n))

reset!(sp::SamplerTag{<:Counts}, n...) = (reset!(sp.data, n...); sp)

function rand(rng::AbstractRNG, sp::SamplerTag{<:Counts{T}}) where T
    dict = Dict{T,Int}()
    for x in rand(rng, sp.data)
        dict[x] = get(dict, x, 0) + 1
    end
    dict
end

"""
    counts(x, [n::Integer])

Equivalent to `rand(Counts(x))`, or to `rand(Counts(Fill(x, n)))`
when `n` is specified.

!!! warning
    Experimental function.
"""
counts(x) = rand(Counts(x))
counts(x, n) = rand(Counts(Fill(x, n)))


## Unique

struct Unique{T,X} <: Distribution{T}
    x::X

    Unique(x::X) where {X} = new{gentype(x),X}(x)
end

Unique(::Type{X}) where {X} = Unique(Uniform(X))

Sampler(RNG::Type{<:AbstractRNG}, u::Unique, n::Val{1}) =
    Sampler(RNG, u.x, n)

Sampler(RNG::Type{<:AbstractRNG}, u::Unique, n::Val{Inf}) =
    SamplerTag{typeof(u)}((x    = Sampler(RNG, u.x, n),
                           seen = Set{gentype(u)}()))

function reset!(sp::SamplerTag{<:Unique}, n=0)
    seen = sp.data.seen
    n > length(seen) && sizehint!(seen, n)
    empty!(seen)
    sp
end

function rand(rng::AbstractRNG, sp::SamplerTag{<:Unique})
    seen = sp.data.seen
    while true
        x = rand(rng, sp.data.x)
        x in seen && continue
        push!(seen, x)
        return x
    end
end
