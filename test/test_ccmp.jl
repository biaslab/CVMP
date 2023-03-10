module CCMPTest

using Test
using ReactiveMP
using CCMP
using Distributions
using StableRNGs
using Flux
using ForwardDiff

@testset "Normal x Normal" begin
    cvi = CVI(StableRNG(42), 1, 2000, Flux.Adam(0.05), ForwardDiffGrad(), 100, Val(true), true)
    
    inbound = NormalMeanVariance()
    outbound = NormalMeanVariance()
    anylytic_x_marginal = prod(ProdAnalytical(), inbound, outbound) 

    q_x = prod(cvi, inbound, outbound)

    @info convert(NormalMeanVariance, q_x)
    
    # q_y = prod(cvi, inbound, outbound, anylytic_x_marginal, (x) -> x)
    # @info convert(NormalMeanVariance, q_y)
end

end
