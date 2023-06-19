import ExponentialFamily
using Flux
using Random

function ReactiveMP.cvi_update!(opt::Flux.Optimise.AbstractOptimiser, λ::AbstractVector, ∇::AbstractVector)
    return Flux.Optimise.update!(opt, λ, ∇)
end

function Base.convert(::Type{ExponentialFamily.KnownExponentialFamilyDistribution}, dist::Distributions.Gamma)
    return ExponentialFamily.KnownExponentialFamilyDistribution(ExponentialFamily.GammaShapeRate, [shape(dist) - one(Float64), -rate(dist)])
end

function Base.convert(::Type{ExponentialFamily.KnownExponentialFamilyDistribution}, dist::ReactiveMP.GammaDistributionsFamily)
    return ExponentialFamily.KnownExponentialFamilyDistribution(ExponentialFamily.GammaShapeRate, [shape(dist) - one(Float64), -rate(dist)])
end

function Random.rand(rng::AbstractRNG, dist::GammaShapeRate, n::Int64)
    return convert(AbstractArray{eltype(dist)}, rand(rng, convert(GammaShapeScale, dist), n))
end

function Distributions.logpdf(exp_dist::ExponentialFamily.KnownExponentialFamilyDistribution{GammaShapeRate}, x)
    η = ExponentialFamily.getnaturalparameters(exponentialfamily_current)
    η1 = first(η)
    η2 = getindex(η, 2)
    return log(x) * η1 + x * η2 - ExponentialFamily.logpartition(exp_dist)
end

function ExponentialFamily.check_valid_natural(::Type{GammaShapeRate}, v::Vector{Float64})
    if length(v) == 2
        return true
    end
    return false
end 

function Base.prod(approximation::CVI, inbound, outbound::GammaDistributionsFamily, in_marginal, nonlinearity)

    @info "called here"

    n_iterations = 1000
    n_gradpoints = 50
    opt = Flux.Adam(0.001)

    benchmark_timings_start = time_ns()

    rng = something(approximation.rng, Random.default_rng())

    # Natural parameters of outbound distribution message
    exponentialfamily_outbound = convert(ExponentialFamily.KnownExponentialFamilyDistribution, convert(Distribution, outbound))

    # Initial parameters of projected distribution
    init_dist = ccmp_init(approximation, inbound, outbound, nonlinearity)
    exponentialfamily_current = convert(ExponentialFamily.KnownExponentialFamilyDistribution, convert(Distribution, init_dist))

    if !ExponentialFamily.isproper(exponentialfamily_current)
        error("Hello from initial distribution")
        return convert(Distribution, exponentialfamily_current)
    end

    # Some distributions implement "sampling" efficient versions
    # returns the same distribution by default
    _, in_marginal_friendly = in_marginal, in_marginal

    hasupdated = false

    samples = ReactiveMP.cvilinearize(rand(rng, in_marginal_friendly, n_gradpoints))
    # compute gradient of log-likelihood
    # the multiplication between two logpdfs is correct
    # we take the derivative with respect to `η`
    # `logpdf(outbound, sample)` does not depend on `η` and is just a simple scalar constant
    weights = map((sample) -> total_derivative(approximation, nonlinearity, sample) * pdf(inbound, sample), samples)
    logq = let samples = samples, weights = weights
        (η) -> mean(
            map(
                (sample, weight) ->
                    weight * logpdf(ExponentialFamily.KnownExponentialFamilyDistribution(ExponentialFamily.GammaShapeRate, η, nothing), nonlinearity(sample...)),
                samples,
                weights
            )
        )
    end

    η = ExponentialFamily.getnaturalparameters(exponentialfamily_current)
    η1 = first(η)
    η2 = getindex(η, 2)

    η_outbound_params = ExponentialFamily.getnaturalparameters(exponentialfamily_outbound)

    if any(isnan.(η))
        error("Hello from isnan init")
    end

    for _ in 1:n_iterations
        ∇logq = ReactiveMP.compute_gradient(approximation.grad, logq, ExponentialFamily.getnaturalparameters(exponentialfamily_current))

        if any(isnan.(ExponentialFamily.getnaturalparameters(exponentialfamily_current)))
            return init_dist
        end

        # compute Fisher matrix 
        Fisher = ExponentialFamily.fisherinformation(exponentialfamily_current)

        ∇f = Fisher \ ∇logq

        # compute gradient on natural parameters
        ∇ = ExponentialFamily.getnaturalparameters(exponentialfamily_current) - ExponentialFamily.getnaturalparameters(exponentialfamily_outbound) - ∇f
        
        # perform gradient descent step
        λ_new = ReactiveMP.cvi_update!(opt, ExponentialFamily.getnaturalparameters(exponentialfamily_current), ∇)
        p_new = ExponentialFamily.KnownExponentialFamilyDistribution(ExponentialFamily.GammaShapeRate, λ_new)
        message_new = ExponentialFamily.KnownExponentialFamilyDistribution(ExponentialFamily.GammaShapeRate, λ_new - η_outbound_params)
        
        # check whether updated natural parameters are proper
        if ExponentialFamily.isproper(p_new) && ExponentialFamily.isproper(message_new)
            exponentialfamily_current = ExponentialFamily.KnownExponentialFamilyDistribution(ExponentialFamily.GammaShapeRate, λ_new, nothing)
            hasupdated = true
        end
    end

    if !hasupdated && approximation.warn
        @warn "CVI approximation has not updated the initial state. The method did not converge. Set `warn = false` to supress this warning."
    end

    benchmark_timings_end = time_ns()

    benchmark_timings[] = benchmark_timings[] + (benchmark_timings_end - benchmark_timings_start)

    η = ExponentialFamily.getnaturalparameters(exponentialfamily_current)
    η1 = first(η)
    η2 = getindex(η, 2)

    if any(isnan.(η))
        return init_dist
    end

    return ReactiveMP.GammaShapeRate(η1 + one(η1), -η2)
end

