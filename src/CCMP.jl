module CCMP

using ReactiveMP
using Random
using Distributions
using SpecialFunctions
import Base: prod
using StableRNGs
using Flux

include("redifinitions.jl")
include("extra_rules.jl")

function Random.rand(rng::AbstractRNG, factorizedjoint::ReactiveMP.FactorizedJoint)
    tuple([rand(rng, dist) for dist in ReactiveMP.getmultipliers(factorizedjoint)]...)
end

function Random.rand(rng::AbstractRNG, factorizedjoint::ReactiveMP.FactorizedJoint, size::Int64)
    [
        rand(rng, factorizedjoint) for _ in 1:size
    ]
end

function Random.rand(factorizedjoint::ReactiveMP.FactorizedJoint, size::Int64)
    rand(Random.GLOBAL_RNG, factorizedjoint, size)
end

function Distributions.pdf(messages::Tuple, x::Tuple)
    prod(map((message_point) -> pdf(message_point[1], message_point[2]), zip(messages, x)))
end

function ccmp_init(_, inbound::GammaDistributionsFamily, outbound::GammaDistributionsFamily, ::typeof(identity))
    return prod(ReactiveMP.ProdAnalytical(), inbound, outbound)
end

function ccmp_init(_, _, outbound, _)
    return outbound
end

function ccmp_init(approximation, inbound, outbound::GaussianDistributionsFamily, nonlinearity)
    samples = rand(approximation.rng, inbound, 100)
    f_samples = map(nonlinearity, samples)
    inbound = NormalMeanVariance(mean(f_samples), var(f_samples))
    return prod(ReactiveMP.ProdAnalytical(), outbound, inbound)
end

function ccmp_init(_, inbound::Tuple, outbound::GaussianDistributionsFamily, nonlinearity)
    s = tuple(map(mean, inbound)...)
    return NormalMeanPrecision(nonlinearity(s...), 1-1/var(outbound))
end

function total_derivative(approximation, f, s::Real)
    return ReactiveMP.compute_derivative(approximation.grad, f, s)
end

function total_derivative(approximation, f, s::Tuple)
    f_for_grad = (x) -> f(x...)
    gradient_at_s = ReactiveMP.compute_gradient(approximation.grad, f_for_grad, collect(s))
    return dot(collect(s), gradient_at_s)
end

function Base.prod(approximation::CVI, inbound, outbound, in_marginal, nonlinearity)
    rng = something(approximation.rng, Random.default_rng())

    # Natural parameters of outbound distribution message
    η_outbound = naturalparams(outbound)

    # Natural parameter type of incoming distribution
    T = typeof(η_outbound)

    # Initial parameters of projected distribution
    init_dist = ccmp_init(approximation, inbound, outbound, nonlinearity)
    λ_current = naturalparams(init_dist)

    if !isproper(λ_current)
        return convert(Distribution, λ_current)
    end

    # Some distributions implement "sampling" efficient versions
    # returns the same distribution by default
    _, in_marginal_friendly = ReactiveMP.logpdf_sample_friendly(in_marginal)

    hasupdated = false

    # @info "total derivative included"
    # @info "total derivative excluded"
    # error(1)

    for _ in 1:(approximation.n_iterations)
        # compute gradient of log-likelihood
        # the multiplication between two logpdfs is correct
        # we take the derivative with respect to `η`
        # `logpdf(outbound, sample)` does not depend on `η` and is just a simple scalar constant
        samples = ReactiveMP.cvilinearize(rand(rng, in_marginal_friendly, approximation.n_gradpoints))

        logq = let samples = samples, inbound = inbound, T = T
            (η) -> mean((sample) -> total_derivative(approximation, nonlinearity, sample) * pdf(inbound, sample) * logpdf(ReactiveMP.as_naturalparams(T, η), nonlinearity(sample...)), samples)
            # (η) -> mean((sample) -> pdf(inbound, sample) * logpdf(ReactiveMP.as_naturalparams(T, η), nonlinearity(sample...)), samples)
        end

        ∇logq = ReactiveMP.compute_gradient(approximation.grad, logq, vec(λ_current))

        # compute Fisher matrix 
        Fisher = ReactiveMP.compute_fisher_matrix(approximation, T, vec(λ_current))

        # compute natural gradient
        ∇f = Fisher \ ∇logq

        # compute gradient on natural parameters
        ∇ = λ_current - η_outbound - as_naturalparams(T, ∇f)

        # perform gradient descent step
        λ_new = as_naturalparams(T, ReactiveMP.cvi_update!(approximation.opt, λ_current, ∇))

        # check whether updated natural parameters are proper
        if isproper(λ_new) && ReactiveMP.enforce_proper_message(approximation.enforce_proper_messages, λ_new, η_outbound)
            λ_current = λ_new
            hasupdated = true
        end
    end

    if !hasupdated && approximation.warn
        @warn "CVI approximation has not updated the initial state. The method did not converge. Set `warn = false` to supress this warning."
    end

    return convert(Distribution, λ_current)
end

function proj(_, dist::GammaDistributionsFamily, exp_dist::GammaDistributionsFamily)
    return dist
end

function proj(approximation::CVI, dist::ContinuousUnivariateLogPdf, exp_dist)
    projected_params = ReactiveMP.naturalparams(ReactiveMP.prod(approximation, dist, exp_dist)) - naturalparams(exp_dist)
    return convert(Distribution, projected_params)
end

end
