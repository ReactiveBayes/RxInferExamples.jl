module GCModel

using RxInfer, LinearAlgebra, Distributions

export gc_car_model, make_constraints

# Linear-Gaussian SSM in generalized coordinates for x = [pos, vel, acc].
# If the observation is 1D, R should be 1x1 and B of size 1x3.
# If the observation is 2D, R should be 2x2 and B of size 2x3.
@model function gc_car_model(y, A, B, Q, R, x0_mean, x0_cov)
    x[1] ~ MvNormal(μ = x0_mean, Σ = x0_cov)
    y[1] ~ MvNormal(μ = B * x[1], Σ = R)
    for t in 2:length(y)
        x[t] ~ MvNormal(μ = A * x[t-1], Σ = Q)
        y[t] ~ MvNormal(μ = B * x[t], Σ = R)
    end
end

"""
make_constraints()
Return a standard mean-field constraint suitable for linear Gaussian models.
"""
function make_constraints()
    return @constraints begin
        q(x, y) = q(x)q(y)
    end
end

end # module
