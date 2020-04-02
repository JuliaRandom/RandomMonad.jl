using Test
using Random: Random, MersenneTwister, rand!, Sampler, randstring
using RandomMonad
using RandomMonad: wrap

@testset "Uniform/Wrap" begin
    for W = (Uniform, wrap)
        for (dist, dest) in (Float64     => x -> 0 <= x < 1,
                             1:10        => 1:10,
                             Int         => typemin(Int):typemax(Int),
                             "asd"       => ['a', 's', 'd'],
                             Set(1:3)    => 1:3,
                             Dict(1=>2)  => [1=>2],
                             (1, 2, 3)   => 1:3)
            w = W(dist)
            if dist isa DataType
                @test w isa RandomMonad.UniformType
            else
                @test w isa RandomMonad.UniformWrap
            end
            @test wrap(w) === w
            if dest isa Function
                @test dest(rand(w))
            else
                @test rand(w) ∈ dest
            end
        end
    end
    s = Sampler(MersenneTwister, 1:3)

    @test_throws MethodError Uniform(1.3)
    @test_throws MethodError Uniform(max)
    @test_throws MethodError Uniform(s)

    @test wrap(1.3) isa RandomMonad.Wrap
    @test wrap(s) === s
    d = Bernoulli()
    @test wrap(d) === d
    @test_throws ArgumentError rand(wrap(1.3))
    @test_throws ArgumentError rand(wrap(max))
end

@testset "Bernoulli" begin
    @test rand(Bernoulli()) ∈ (0, 1)
    @test rand(Bernoulli()) isa Bool
    for T = (Int, UInt, Bool)
        @test rand(Bernoulli(T)) ∈ (0, 1)
    end
    @test rand(Bernoulli(1)) == 1
    @test rand(Bernoulli(0)) == 0
    # TODO: do the math to estimate proba of failure:
    @test 620 < count(rand(Bernoulli(Bool, 0.7), 1000)) < 780
    for T = (Bool, Int, Float64, ComplexF64)
        r = rand(Bernoulli(T))
        @test r isa T
        @test r ∈ (0, 1)
        r = rand(Bernoulli(T, 1))
        @test r == 1
    end
end

@testset "Binomial" begin
    b = Binomial(1000, 0.7)
    @test rand(b) isa Int
    @test eltype(b) == Int
    @test 620 < rand(b) < 780
    b = Binomial(1000)
    @test b.p == 0.5
    @test 420 < rand(b) < 580
    b = Binomial()
    @test b.n == 1
    @test rand(b) ∈ 0:1
end

@testset "Categorical" begin
    n = rand(1:9)
    @test rand(Categorical(n)) ∈ 1:9
    @test all(∈(1:9), rand(Categorical(n), 10))
    @test rand(Categorical(n)) isa Int
    c = Categorical(Float64(n))
    @test rand(c) isa Float64
    @test rand(c) ∈ 1:9
    c = Categorical([1, 7, 2])
    # cf. Bernoulli tests
    @test 620 < count(==(2), rand(c, 1000)) < 780
    @test rand(c) isa Int
    @test rand(Categorical{Float64}((1, 2, 3, 4))) isa Float64

    c = Categorical(3)
    @test Categorical{Float64}(c) isa Categorical{Float64}
    @test convert(Categorical{Float64}, c) isa Categorical{Float64}
    @test_throws ArgumentError convert(Categorical{Float64}, 3)

    @test_throws ArgumentError Categorical(())
    @test_throws ArgumentError Categorical([])
    @test_throws ArgumentError Categorical(x for x in 1:0)
end

@testset "Multinomial" begin
    m = Multinomial(Categorical(3), 10)
    @test rand(m) isa Vector{Int}
    for a in rand(m, 3)
        @test sum(a) == 10
    end
    m = Multinomial(Categorical([1, 0, 1]), 10)
    a = rand(m)
    @test a[2] == 0
    @test a[1] + a[3] == 10
end

@testset "MixtureModel" begin
    m = MixtureModel([100:300, Normal(), Uniform(500:1000)], [1, 7, 2])
    x = rand(m)
    testtype(x) = x isa Float64 || x isa Int
    @test testtype(x)
    xs = rand(m, 1000)
    @test all(testtype, xs)
    # cf. Bernoulli tests
    @test 620 < count(x -> x isa Float64, xs) < 780

    m = MixtureModel([Normal(), CloseOpen()], (1,2))
    @test eltype(m.components) == Distribution{Float64} # not crucial

    m = MixtureModel([Normal(), CloseOpen()])
    @test m.prior.cdf == [0.5, 1.0]
    # test that the first argument doesn't need to have length defined
    m = MixtureModel(Iterators.takewhile(_->true, (Normal(), Exponential(), CloseOpen())))
    @test m.prior.cdf == [1/3, 2/3, 1.0]

    @test_throws ArgumentError MixtureModel([1:3, Normal()], [1, 2, 3])
end

@testset "Normal" begin
    @test rand(Normal()) isa Float64
    @test rand(Normal(0.0, 1.0)) isa Float64
    @test rand(Normal(0, 1)) isa Float64
    @test rand(Normal(0, 1.0)) isa Float64
    @test rand(Normal(Float32)) isa Float32
    @test rand(Normal(ComplexF64)) isa ComplexF64
end

@testset "Exponential" begin
    @test rand(Exponential()) isa Float64
    @test rand(Exponential(1.0)) isa Float64
    @test rand(Exponential(1)) isa Float64
    @test rand(Exponential(Float32)) isa Float32
end

@testset "Poisson" begin
    p = Poisson()
    @test p.λ == 1
    @test rand(p) isa Int
    @test rand(Poisson(2)) isa Int
    @test rand(Poisson(3.0)) isa Int
    @test rand(Poisson(0)) == 0
end

@testset "rand(::AbstractFloat)" begin
    # check that overridden methods still work
    m = MersenneTwister()
    for F in (Float16, Float32, Float64, BigFloat)
        @test rand(F) isa F
        sp = Random.Sampler(MersenneTwister, RandomMonad.CloseOpen01(F))
        @test rand(m, sp) isa F
        @test 0 <= rand(m, sp) < 1
        for (CO, (l, r)) = (CloseOpen  => (<=, <),
                            CloseClose => (<=, <=),
                            OpenOpen   => (<,  <),
                            OpenClose  => (<,  <=))
            f = rand(CO(F))
            @test f isa F
            @test l(0, f) && r(f, 1)
        end
        F ∈ (Float64, BigFloat) || continue # only types implemented in Random
        sp = Random.Sampler(MersenneTwister, RandomMonad.CloseOpen12(F))
        @test rand(m, sp) isa F
        @test 1 <= rand(m, sp) < 2
    end
    @test CloseOpen(1,   2)          === CloseOpen(1.0, 2.0)
    @test CloseOpen(1.0, 2)          === CloseOpen(1.0, 2.0)
    @test CloseOpen(1,   2.0)        === CloseOpen(1.0, 2.0)
    @test CloseOpen(1.0, Float32(2)) === CloseOpen(1.0, 2.0)
    @test CloseOpen(big(1), 2) isa CloseOpen{BigFloat}

    for CO in (CloseOpen, CloseClose, OpenOpen, OpenClose)
        @test_throws ArgumentError CO(1, 1)
        @test_throws ArgumentError CO(2, 1)

        @test CO(Float16(1), 2) isa CO{Float16}
        @test CO(1, Float32(2)) isa CO{Float32}
    end
end


## adapters

@testset "Const" begin
    @test rand(Const(1)) === 1
    @test rand(Const(1:3)) === 1:3
    @test eltype(Const(1:3)) == UnitRange{Int}
    s = Set([1, 2, 3])
    @test rand(Const(s)) === s
    a = rand(Const(s), 9)
    @test eltype(a) == Set{Int}
    @test all(x -> x === s, a)
end

@testset "algebra" begin
    d = Uniform(Float64) + Uniform(1:3)
    @test eltype(d) == Float64
    @test all(x -> 1.0 <= x < 4, rand(d, 100))
    for op = (+, -, *, ^, /)
        # Filter below for y^x needing x >= 0
        for x = (Filter(x -> x >= 0, Uniform(Int32)), Int32(3))
            for d = (op(x, Uniform(Int32(1):Int32(3))),
                     op(Uniform(Int32(1):Int32(3)), x))
                if op == /
                    @test eltype(d) == Float64
                    @test rand(d) isa Float64
                else
                    @test eltype(d) == Int32
                    @test rand(d) isa Int32
                end
            end
        end
    end
    # test reset!
    @test all(==(1:9), sort!.(rand(Fill(Unique(1:9) + Uniform(0:0), 9), 2)))

    # getindex
    g = Const('a':'z')[Categorical(3)]
    @test rand(g) ∈ 'a':'c'
    g = Const(Dict(1=>3, 2=>4))[Uniform(1:2)]
    @test rand(g) ∈ 3:4
end

@testset "Bind" begin
    b = Bind(n -> Fill(Bool, n), 1:5)
    @test eltype(b) == Vector{Bool}
    @test rand(b) isa Vector{Bool}
    for v in rand(b, 4)
        @test v isa Vector{Bool}
        @test length(v) in 1:5
    end

    b = Bind(Zip([Bool, Normal(), 1:9], Shuffle(1:4))) do (t, n)
               Fill(t, n)
           end
    vs = rand(b, 4)
    @test allunique(length.(vs))
    @test all(v -> length(v) ∈ 1:4, vs)
    @test all(v -> eltype(v) ∈ [Bool, Float64, Int], vs)
end

@testset "Lift" begin
    d = Lift(x -> 2x, 1:3)
    @test rand(d) ∈ 2:2:6
    @test all(x -> x ∈ 2:2:6, rand(d, 100))
    @test eltype(d) == Int
    @test rand(d) isa Int

    d = Lift{Float64}(x -> x > 0, Normal())
    @test rand(d) isa Float64
    @test all(x -> x ∈ (0.0, 1.0), rand(d, 100))

    d = Lift(+, Float64, [10, 20])
    @test all(rand(d, 30)) do x
        10 <= x < 11 || 20 <= x < 21
    end
end

@testset "Map" begin
    d = Map(x -> 2x, Fill(1:3, 3))
    @test all(∈(2:2:6), rand(d))
    @test all(x -> all(∈(2:2:6), x), rand(d, 10))
    @test eltype(d) == Vector{Int}
    @test rand(d) isa Vector{Int}

    d = Map{Vector{Float64}}(x -> x > 0, Fill(Normal(), 10))
    @test rand(d) isa Vector{Float64}
    @test all(x -> all(∈((0.0, 1.0)), x), rand(d, 5))

    d = Map(+, Fill(Float64, 2), Fill([10, 20], 2))
    @test all(rand(d)) do x
        10 <= x < 11 || 20 <= x < 21
    end
end

@testset "Filter" begin
    d = Filter(x -> x > 0, Normal())
    @test all(x -> x > 0, rand(d, 1000))
    d = Filter(x -> x < 4, Categorical([1, 7, 2, 10]))
    # cf. Bernoulli tests
    @test 620 < count(==(2), rand(d, 1000)) < 780
    d = Filter(x -> x < 9, 1:20)
    @test all(x -> 1 <= x <= 9, rand(d, 1000))

    # reset!
    s = Random.Sampler(MersenneTwister, Filter(x -> x < 5, Shuffle(1:9)))
    for _ = 1:3
        a = rand(s, 4)
        @test allunique(a)
        @test all(x -> x < 5, a)
    end
end

@testset "Reduce" begin
    r = Reduce(+, Fill(1:3, 2))
    @test rand(r) ∈ 2:6
    @test eltype(r) == Int
    @test all(x -> x==9, rand(Reduce(+, Zip((8,), (1,))), 10))
    f = Fill(Reduce(+, Unique(1:4)), 2)
    @test rand(f) isa Vector{Int}
    # check that reset! works properly (otherwise, rand(f, 3) would never terminate)
    v = rand(f, 3)
    @test v isa Vector{Vector{Int}}
    @test all(allunique, v)
end

@testset "Counts" begin
    c = Counts(Fill(0x1:0x3, 100))
    @test eltype(c) == Dict{UInt8,Int}
    d = rand(c)
    @test d isa Dict{UInt8,Int}
    @test keys(d) ⊆ 1:3
    @test 100 <= sum(values(d)) <= 300
    # check reset! works
    vs = rand(Fill(Counts(Unique(1:3)), 3), 3)
    @test vs isa Vector{Vector{Dict{Int,Int}}}
    for v in vs
        ks = []
        for d in v
            append!(ks, keys(d))
            @test collect(values(d)) == 1:1
        end
        @test sort!(ks) == 1:3
    end
end

@testset "Unique" begin
    u = Unique(1:3)
    @test eltype(u) == Int
    @test rand(u) ∈ 1:3
    a = rand(u, 3)::Vector{Int}
    @test allunique(a)

    z = Fill(Zip(u, u), 3)
    for i=1:3
        rz = rand(z)
        for a = (rand(Fill(u, 3)), first.(rz), last.(rz),
                 rand(Fill(Lift(identity, u), 3)),
                 rand(Fill(Lift(+, (0,), u), 3)),
                 rand(Filter(x->true, u)))
            @test all(in(1:3), a)
            @test allunique(a)
        end
    end

    u = Unique(Bool)
    @test allunique(rand(u, 2))
end

@testset "$ShuffleAlgo" for ShuffleAlgo = (FisherYates, SelfAvoid, Shuffle)
    u = ShuffleAlgo(1:3)
    @test eltype(u) == Int
    @test rand(u) ∈ 1:3
    a = rand(u, 3)::Vector{Int}
    @test allunique(a)

    f = Fill(u, 3)
    z = Fill(Zip(u, u), 3)
    for i=1:3
        rz = rand(z)
        for a = (rand(f), first.(rz), last.(rz))
            @test all(in(1:3), a)
            @test allunique(a)
        end
    end

    u = ShuffleAlgo('a':'z')
    @test rand(u) isa Char
    @test allunique(rand(u, 26))

    # test reset!
    a = rand(Fill(Fill(ShuffleAlgo(1:9), 5), 2))
    @test a isa Vector{Vector{Int}}
    @test !allunique(vcat(a...))

    # test reset!(sp, 1) works
    @test all(in(1:4), vcat(rand(Fill(ShuffleAlgo(1:4), 1), 10)...))
    # test reset!(sp, 2) works
    @test all(rand(Fill(ShuffleAlgo(1:4), 2), 100)) do x
        x[1] != x[2]
    end

    # test Val(1)-sampler works
    s = Random.Sampler(MersenneTwister, ShuffleAlgo(1:9), Val(1))
    @test allunique(rand(s, 9))

    # old bug with FisherYates
    @test counts(Lift(Fill(Fill(ShuffleAlgo(1:9), 1), 4)) do xs
                      for i = 1:length(xs[1])
                          all(x -> x[i] == xs[1][i], xs) && return true
                      end
                      false
                 end,
                 1000)[false] > 990 # would fail probably less than once in 30000

    # compatibility with rand!
    s = Random.Sampler(MersenneTwister, ShuffleAlgo(1:4))
    # when rand(s, 4) is called in a loop, we test that `reset!(s, 4)`
    # is called each time; otherwise, `s` would get exhausted after
    # the first call and an exception would be thrown
    for _ = 1:3
        a = rand(s, 4)
        @test all(in(1:4), a)
        @test allunique(a)
    end

    # multivariate
    if ShuffleAlgo == Shuffle
        a = Char[]

        s = Shuffle(['a'], 0)
        @test rand(s) == Char[]
        @test rand!(a, s, Val(1)) == Char[]

        s = Shuffle(['a'], 1)
        @test rand(s) == ['a']
        @test rand!(a, s, Val(1)) == ['a']

        s = Shuffle(['a', 'b'], 2)
        @test sort!(rand(s)) == ['a', 'b']
        @test sort!(rand!(a, s, Val(1))) == ['a', 'b']

        s = Shuffle(['a', 'b', 'c'], 3)
        @test sort!(rand(s)) == ['a', 'b', 'c']
        @test sort!(rand!(a, s, Val(1))) == ['a', 'b', 'c']

        s = Shuffle(collect(randstring(100)), 50) # FisherYates
        v = rand(s)
        @test v isa Vector{Char}
        @test length(v) == 50
        @test rand!(a, s, Val(1)) === a
        @test length(a) == 50

        s = Shuffle(collect(randstring(1000)), 2) # SelfAvoid
        v = rand(s)
        @test v isa Vector{Char}
        @test length(v) == 2
        @test rand!(a, s, Val(1)) === a
        @test length(a) == 2
    end
end

## containers

@testset "Zip" begin
    z = Zip(Normal(), 1:3)
    @test eltype(z) == Tuple{Float64,Int}
    @test rand(z) isa Tuple{Float64,Int}
    @test all(x -> x ∈ 1:3, last.(rand(z, 100)))

    z = Zip()
    @test rand(z) == ()
    z = Zip(1:3)
    @test rand(z) isa Tuple{Int}
    @test rand(z)[1] in 1:3
    z = Zip(1:2, Normal(), Int8)
    @test rand(z) isa Tuple{Int,Float64,Int8}
    @test all(x -> x isa Tuple{Int,Float64,Int8}, rand(z, 9))

    # Zip(::Sampler...)
    s = Sampler(MersenneTwister, 1:3)
    z = Zip(s, s)
    @test rand(z) isa Tuple{Int,Int}
    @test all(in(1:3), rand(z))
    for t in rand(z, 3)
        @test all(in(1:3), rand(z))
    end

    # recursive rand!
    z1 = Zip(Fill(1:3, 2), Fill(1:4, 3))
    z2 = Zip(Fill(4:6, 2), Fill(5:8, 3))
    t = rand(z1)
    a, b = t[1][1], t[2][1]
    rand!(t, z2, Val(2))
    @test t[1][1] != a
    @test t[2][1] != b
    @test all(∈(5:8), t[2])
    rand!([t], Fill(z1, 1), Val(3))
    @test all(∈(1:4), t[2])
    @test_throws ArgumentError rand!(t, Zip(1:2, 1:3), Val(1)) # can't mutate tuple
    @test_throws ArgumentError rand!((t..., t...), z2, Val(2)) # not same size
end

@testset "Fill" begin
    rng = MersenneTwister()
    for f in (Fill(1:9, (2, 3)),
              Fill(Uniform(1:9), 2, 3))
        @test f isa Distribution{Array{Int,2}}
        a = rand(f)
        b = similar(a)
        rand!(rng, b, f, Val(1))
        c = similar(a)
        rand!(c, f, Val(1))
        for A = (a, b, c)
            @test A isa Array{Int,2}
            @test size(A) == (2, 3)
            @test all(x -> x ∈ 1:9, A)
        end
        @test_throws ArgumentError rand!(similar(a, 3, 2), f, Val(1))
        @test_throws ArgumentError rand!(similar(a, 6), f, Val(1))
    end
    f = Fill(Float64)
    @test rand(f) isa Array{Float64,0}
    f = Fill(Float64, 0x2, Int128(3))
    a = rand(f)
    @test a isa Array{Float64,2}
    @test size(a) == (2, 3)
    f = Fill(1:9, 9)
    a = rand!(Int8[], f, Val(1))
    @test a isa Vector{Int8}
    @test length(a) == 9
    @test all(in(1:9), a)

    # Fill(::Sampler, ...)
    s = Sampler(MersenneTwister, 1:3)
    f = Fill(s, 2)
    @test all(in(1:3), rand(f))

    # recursive rand!
    inner = Fill(1:3, 3)
    a = rand(Fill(inner, 4))
    b = copy(a)
    rand!(a, inner)
    @test a[1] !== b[1]
    copy!(b, a)
    rand!(a, Fill(inner, 4), Val(1))
    @test a[1] !== b[1]
    copy!(b, a)
    rand!(a, Fill(inner, 4), Val(2))
    @test a[1] === b[1]
end
