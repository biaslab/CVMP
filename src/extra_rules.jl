# Message(Univariate, Gamma, a=unsafeMean(dist_a)+1, b=unsafeMean(dist_out))
import DomainSets

@rule GammaShapeRate(:α, Marginalisation) (q_out::GammaShapeRate, q_β::GammaShapeRate) = begin
    return ContinuousUnivariateLogPdf(DomainSets.HalfLine(), (α) -> α * mean(log, q_β) + (α - 1) * mean(log, q_out) - loggamma(α))
end

@rule GammaShapeRate(:β, Marginalisation) (q_out::Gamma, q_α::PointMass, ) = begin 
    return GammaShapeRate(mean(q_α)+1, mean(q_out))
end

@rule GammaShapeRate(:β, Marginalisation) (q_out::Gamma, q_α::Gamma, ) = begin 
    return GammaShapeRate(mean(q_α)+1, mean(q_out))
end

@rule GammaShapeRate(:β, Marginalisation) (q_out::GammaShapeRate, q_α::PointMass, ) = begin 
    return GammaShapeRate(mean(q_α)+1, mean(q_out))
end

@rule GammaShapeRate(:β, Marginalisation) (q_out::GammaShapeRate, q_α::SampleList, ) = begin 
    return GammaShapeRate(mean(q_α)+1, mean(q_out))
end

@rule GammaShapeRate(:out, Marginalisation) (q_α::Any, q_β::Any, ) = begin 
    return GammaShapeRate(mean(q_α), mean(q_β))
end