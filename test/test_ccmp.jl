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
        cvi = CVI(StableRNG(42), 1, 2000, Flux.Adam(0.001), ForwardDiffGrad(), 2000, Val(true), true)

        inbound = NormalMeanVariance(2, 10)
        outbound = NormalMeanVariance(4, 10)
        n_analytical = prod(ProdAnalytical(), inbound, outbound)
        q_y = prod(cvi, inbound, outbound, prod(ProdAnalytical(), inbound, outbound), (x) -> x)
        @test prod(cvi, inbound, outbound, prod(ProdAnalytical(), inbound, outbound), (x) -> x) ≈ n_analytical atol = 3e-1
    end

    @testset "Bernoulli x Bernoulli" begin
        cvi = CVI(StableRNG(42), 1, 2000, Flux.Adam(0.001), ForwardDiffGrad(), 2000, Val(true), true)

        inbound = Bernoulli(0.5)
        outbound = Bernoulli(0.6)
        n_analytical = prod(ProdAnalytical(), inbound, outbound)
        q_y = prod(cvi, inbound, outbound, prod(ProdAnalytical(), inbound, outbound), (x) -> x)
        @info prod(cvi, inbound, outbound, prod(ProdAnalytical(), inbound, outbound), (x) -> x)
        @test prod(cvi, inbound, outbound, prod(ProdAnalytical(), inbound, outbound), (x) -> x) ≈ n_analytical atol = 3e-1
    end
end
end
