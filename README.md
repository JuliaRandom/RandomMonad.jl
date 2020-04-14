# RandomMonad

[![Build Status](https://travis-ci.org/rfourquet/RandomMonad.jl.svg?branch=master)](https://travis-ci.org/rfourquet/RandomMonad.jl)

RandomMonad provides a number of composable primitives for constructing
"distributions". A distribution is understood in a broad sense: it is anything
on which `rand` can be called. A distribution is like a recipe describing how
to construct an object of a certain type. For example, `1:3` is an implicit
distribution describing how to pick randomly an `Int` among `1`, `2`, `3`.

Unlike [Distributions.jl](https://github.com/JuliaStats/Distributions.jl)
which seriously addresses mathematical needs, the `RandomMonad` package is
less specific and is intended to be generally useful for implementing
randomness. This is reflected in the core type, a simple `Distribution{T}`,
where `T` can be anything and is just the type of generated values.

The scope of the package is to provide ready-made tools which can be easily
combined to create on-the-fly distributions. On the other hand, the
[RandomExtensions.jl](https://github.com/rfourquet/RandomExtensions.jl)
package can assist in the definition of sampling methods for custom objects in
library code.

Currently, `RandomMonad` also implements few classical mathematical
distributions, like `Bernoulli` or `Poisson`, but these might eventually be
split off in another dedicated package.

## Examples

Perhaps one of the simplest ways to combine distributions is to use arithmetic
operators on them. In the following example, `Uniform(-1:1)` makes `-1:1` an
explicit distribution (subtype of `Distribution`), which is required in this
case:

```julia-repl
julia> rand(5 * Uniform(-1:1) + Normal(), 6)
6-element Array{Float64,1}:
  0.3747282015594068
 -4.5476618093190115
  4.193624739643829
  6.758516580679736
  0.6079472623893868
 -0.5200061992409745
```

As one might guess, this takes a random integer in `[-1, 1]`, multiplies it by
`5`, and adds the result to another random number drawn from the normal
distribution.

A basic distribution is `Fill(d, n)`, defining the generation of arrays of
length `n` of elements drawn from distribution `d`:

```julia-repl
julia> f = Fill(1:9, 4)
Fill(1:9, 4)

julia> eltype(f)
Array{Int64,1}

julia> rand(f)
4-element Array{Int64,1}:
 8
 3
 8
 4
```

Many algorithms which use randomness can be encapsulated as a distribution.
For example, `Shuffle` defines an alternate API to the `Random.shuffle` function,
but is more general. The following example creates an array of two vectors of
length `4` of distinct elements from `1:5`:

```julia-repl
julia> rand(Fill(Shuffle(1:5), 4), 2)
2-element Array{Array{Int64,1},1}:
 [3, 4, 5, 1]
 [4, 1, 2, 5]
```
This is sampling from a collection "without replacement", and is
equivalent to `[StatsBase.sample(1:5, 4, replace=false) for _=1:2]`.

## But... what is a Monad??

I won't add yet another tutorial on monads, but the good news is that knowing
the theory of monads is not at all necessary for using this package. It just
so happens that `Distributions{T}` has a monadic structure, and the package
provides some related "combinators" (basic blocs to create more elaborate
constructions).
