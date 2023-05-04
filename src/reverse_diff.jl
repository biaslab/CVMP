using ReverseDiff
using LogDensityProblemsAD
export ReverseDiffGrad

struct ReverseDiffGrad end

# const RDCache = Ref(true)

# getrdcache() = RDCache[]

# setrdcache(b::Bool) = setrdcache(Val(b))
# setrdcache(::Val{false}) = RDCache[] = false
# setrdcache(::Val{true}) = RDCache[] = true

# for cache in (:true, :false)
#     @eval begin
#         function LogDensityProblemsAD.ADgradient(::ReverseDiffAD{$cache}, ℓ::Turing.LogDensityFunction)
#             return LogDensityProblemsAD.ADgradient(Val(:ReverseDiff), ℓ; compile=Val($cache))
#         end
#     end
# end

ReactiveMP.compute_derivative(::ReverseDiffGrad, f::F, value::Real) where {F}       = ReverseDiff.gradient((x) -> f(x[1]), [value])[1]
ReactiveMP.compute_gradient(::ReverseDiffGrad, f::F, vec::AbstractVector) where {F} = ReverseDiff.gradient(f, vec)
ReactiveMP.compute_hessian(::ReverseDiffGrad, f::F, vec::AbstractVector) where {F}  = ReverseDiff.hessian(f, vec)