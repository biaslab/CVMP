module CCMPTest

using Test
using ReactiveMP
using CCMP
using Distributions
using StableRNGs
using Flux
using ForwardDiff

@testset "ccmp prod tests" begin
    @testset "Normal x Normal" begin
        cvi = CVI(StableRNG(42), 1, 2000, Flux.Adam(0.001), ForwardDiffGrad(), 10, Val(true), true)

        inbound = NormalMeanVariance(2, 10)
        outbound = NormalMeanVariance(4, 10)
        n_analytical = prod(ProdAnalytical(), inbound, outbound)
        q_y = prod(cvi, inbound, outbound, prod(ProdAnalytical(), inbound, outbound), (x) -> x)
        @info q_y
        @test prod(cvi, inbound, outbound, prod(ProdAnalytical(), inbound, outbound), (x) -> x) ≈ n_analytical atol = 3e-1
    end

    @testset "Bernoulli x Bernoulli" begin
        cvi = CVI(StableRNG(42), 1, 2000, Flux.Adam(0.001), ForwardDiffGrad(), 10, Val(true), true)

        inbound = Bernoulli(0.5)
        outbound = Bernoulli(0.6)
        n_analytical = prod(ProdAnalytical(), inbound, outbound)
        q_y = prod(cvi, inbound, outbound, prod(ProdAnalytical(), inbound, outbound), (x) -> x)
        @info prod(cvi, inbound, outbound, prod(ProdAnalytical(), inbound, outbound), (x) -> x)
        @test prod(cvi, inbound, outbound, prod(ProdAnalytical(), inbound, outbound), (x) -> x) ≈ n_analytical atol = 3e-1
    end

    @testset "Gamma x Gamma" begin
        cvi = CVI(StableRNG(42), 1, 100, Flux.Adam(0.001), ForwardDiffGrad(), 100, Val(true), true)

        for i = 1:10, j = 1:10, k = 1:10, l = 1:10
            inbound = Gamma(i, j)
            outbound = Gamma(k, l)
            n_analytical = convert(ReactiveMP.GammaShapeRate, prod(ProdAnalytical(), inbound, outbound))
            q_y = prod(cvi, inbound, outbound, prod(ProdAnalytical(), inbound, outbound), (x) -> x)
            @info n_analytical
            @info prod(cvi, inbound, outbound, prod(ProdAnalytical(), inbound, outbound), (x) -> x)
            @test prod(cvi, inbound, outbound, prod(ProdAnalytical(), inbound, outbound), (x) -> x) ≈ n_analytical atol = 0.2
        end
    end
end
end
