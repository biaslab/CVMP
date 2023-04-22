@rule GammaShapeRate(:β, Marginalisation) (q_out::Gamma, q_α::PointMass, ) = begin 
    return GammaShapeRate(1000, 1)
end

@rule GammaShapeRate(:β, Marginalisation) (q_out::Gamma, q_α::Gamma, ) = begin 
    return GammaShapeRate(1000, 1)
end

@rule GammaShapeRate(:β, Marginalisation) (q_out::GammaShapeRate, q_α::PointMass, ) = begin 
    return GammaShapeRate(1000, 1)
end

@rule GammaShapeRate(:β, Marginalisation) (q_out::GammaShapeRate, q_α::GammaShapeRate, ) = begin 
    return GammaShapeRate(1000, 1)
end

@rule GammaShapeRate(:α, Marginalisation) (q_out::Gamma, q_β::GammaShapeRate, ) = begin 
    return GammaShapeRate(1000, 1)
end

@rule GammaShapeRate(:α, Marginalisation) (q_out::GammaShapeRate, q_β::GammaShapeRate, ) = begin 
    return GammaShapeRate(1000, 1)
end

@rule GammaShapeRate(:out, Marginalisation) (q_α::PointMass, q_β::GammaShapeRate, ) = begin 
    return GammaShapeRate(mean(q_α), mean(q_β))
end

@rule GammaShapeRate(:out, Marginalisation) (q_α::GammaShapeRate, q_β::GammaShapeRate, ) = begin 
    return GammaShapeRate(mean(q_α), mean(q_β))
end

# @rule GammaShapeRate(:β, Marginalisation) (q_out::Any, q_α::Any, ) = begin 
#     return GammaShapeRate(1000, 1)
# end

# @rule GammaShapeRate(:out, Marginalisation) (q_α::Any, q_β::Any, ) = begin 
#     return GammaShapeRate(mean(q_α), mean(q_β))
# end

# @rule GammaShapeRate(:α, Marginalisation) (q_out::Any, q_β::Any, ) = begin 
#     return GammaShapeRate(1000, 1)
# end