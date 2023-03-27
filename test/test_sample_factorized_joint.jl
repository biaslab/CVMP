module CCMPTest

using Test
using ReactiveMP
using CCMP
using Distributions
using StableRNGs
using Flux
using ForwardDiff
using Random

@testset "rand factorized joint" begin
    fc = ReactiveMP.FactorizedJoint((PointMass(2), PointMass(3)))
    @test rand(fc) == (2, 3)
    @test rand(fc, 2) == [(2, 3), (2, 3)]
end
end
