using Test
using Random: Random, MersenneTwister
using DistributionsLite

@testset "Uniform" begin
    @test rand(Uniform(Float64)) isa Float64
    @test rand(Uniform(1:10)) isa Int
    @test rand(Uniform(1:10)) ∈ 1:10
    @test rand(Uniform(Int)) isa Int
end

@testset "Bernoulli" begin
    @test rand(Bernoulli()) ∈ (0, 1)
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
        sp = Random.Sampler(MersenneTwister, DistributionsLite.CloseOpen01(F))
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
        sp = Random.Sampler(MersenneTwister, DistributionsLite.CloseOpen12(F))
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

@testset "Filter" begin
    d = Filter(x -> x > 0, Normal())
    @test all(x -> x > 0, rand(d, 1000))
    d = Filter(x -> x < 4, Categorical([1, 7, 2, 10]))
    # cf. Bernoulli tests
    @test 620 < count(==(2), rand(d, 1000)) < 780
    d = Filter(x -> x < 9, 1:20)
    @test all(x -> 1 <= x <= 9, rand(d, 1000))
end

@testset "Map" begin
    d = Map(x -> 2x, 1:3)
    @test rand(d) ∈ 2:2:6
    @test all(x -> x ∈ 2:2:6, rand(d, 100))
    @test eltype(d) == Int
    @test rand(d) isa Int

    d = Map{Float64}(x -> x > 0, Normal())
    @test rand(d) isa Float64
    @test all(x -> x ∈ (0.0, 1.0), rand(d, 100))
end

@testset "Unique" begin
    u = Unique(1:3)
    @test eltype(u) == Int
    @test rand(u) ∈ 1:3
    a = rand(u, 3)::Vector{Int}
    @test allunique(a)

    u = Unique(Bool)
    @test allunique(rand(u, 2))
end

@testset "FisherYates" begin
    u = FisherYates(1:3)
    @test eltype(u) == Int
    @test rand(u) ∈ 1:3
    a = rand(u, 3)::Vector{Int}
    @test allunique(a)

    u = FisherYates('a':'z')
    @test rand(u) isa Char
    @test allunique(rand(u, 26))
end

## containers

@testset "Zip" begin
    z = Zip(Normal(), 1:3)
    @test eltype(z) == Tuple{Float64,Int}
    @test rand(z) isa Tuple{Float64,Int}
    @test all(x -> x ∈ 1:3, last.(rand(z, 100)))
end

@testset "Fill" begin
    for f in (Fill(1:9, (2, 3)),
              Fill(Uniform(1:9), 2, 3))
        @test f isa Distribution{Array{Int,2}}
        a = rand(f)
        @test a isa Array{Int,2}
        @test size(a) == (2, 3)
        @test all(x -> x ∈ 1:9, a)
    end
    f = Fill(Float64)
    @test rand(f) isa Array{Float64,0}
    f = Fill(Float64, 0x2, Int128(3))
    a = rand(f)
    @test a isa Array{Float64,2}
    @test size(a) == (2, 3)
end
