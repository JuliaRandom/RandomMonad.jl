module RandomMonad

export Distribution


"""
    Distribution{T}

An instance of a subtype of `Distribution{T}` is an object able
of produce random values of type `T` via `rand`-related functions.
"""
abstract type Distribution{T} end


end # module
