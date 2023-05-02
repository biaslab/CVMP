module CCMPTest

using Test
using ReactiveMP
using CCMP
using Distributions
using StableRNGs
using Flux
using ForwardDiff
using ExponentialFamily
using StatsFuns

@testset "Gamma softmax natural gradient step" begin
    cvi = CVI(StableRNG(42), 1, 2000, Flux.Adam(0.001), ForwardDiffGrad(), 10, Val(true), true)

    samples = [86.2027941354432, 88.01191974410457]
    point = ExponentialFamily.KnownExponentialFamilyDistribution(ExponentialFamily.GammaShapeRate, [258.33366162357294, -4.665785943089478], nothing)
    inbound = ReactiveMP.GammaShapeRate(257.37489915581654, 3.0)
    nonlinearity = (x) -> StatsFuns.softplus(x)
    logq = let samples = samples, inbound = inbound
        (η) -> mean(
            (sample) ->
                CCMP.total_derivative(cvi, nonlinearity, sample) *
                pdf(inbound, sample) *
                logpdf(ExponentialFamily.KnownExponentialFamilyDistribution(ExponentialFamily.GammaShapeRate, η, nothing), nonlinearity(sample...)),
            samples
        )
        # (η) -> mean((sample) -> pdf(inbound, sample) * logpdf(ReactiveMP.as_naturalparams(T, η), nonlinearity(sample...)), samples)
    end

    @info CCMP.natural_gradient_step(cvi, point, logq)
end

end
