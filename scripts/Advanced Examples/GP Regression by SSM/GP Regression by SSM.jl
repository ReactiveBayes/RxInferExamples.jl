# This file was automatically generated from examples/Advanced Examples/GP Regression by SSM/GP Regression by SSM.ipynb
# by notebooks_to_scripts.jl at 2025-03-14T05:52:01.860
#
# Source notebook: GP Regression by SSM.ipynb

using RxInfer, Random, Distributions, LinearAlgebra, Plots

@model function gp_regression(y, P, A, Q, H, var_noise)
    f_prev ~ MvNormal(μ = zeros(length(H)), Σ = P) #initial state
    for i in eachindex(y)
        f[i] ~ MvNormal(μ = A[i] * f_prev,Σ = Q[i])
        y[i] ~ Normal(μ = dot(H , f[i]), var = var_noise)
        f_prev = f[i]
    end
end

Random.seed!(10)
n = 100
σ²_noise = 0.04;
t = collect(range(-2, 2, length=n)); #timeline
f_true = sinc.(t); # true process
f_noisy = f_true + sqrt(σ²_noise)*randn(n); #noisy process

pos = sort(randperm(75)[1:2:75]); 
t_obser = t[pos]; # time where we observe data

y_data = Array{Union{Float64,Missing}}(missing, n)
for i in pos 
    y_data[i] = f_noisy[i]
end

θ = [1., 1.]; # store [l, σ²]
Δt = [t[1]]; # time difference
append!(Δt, t[2:end] - t[1:end-1]);

plot(t, f_true, label="True process f(t)")
scatter!(t_obser, y_data[pos], label = "Noisy observations")
xlabel!("t")
ylabel!("f(t)")

λ = sqrt(3)/θ[1];
#### compute matrices for the state-space model ######
L = [0., 1.];
H = [1., 0.];
F = [0. 1.; -λ^2 -2λ]
P∞ = [θ[2] 0.; 0. (λ^2*θ[2]) ]
A = [exp(F * i) for i in Δt]; 
Q = [P∞ - i*P∞*i' for i in A];

result_32 = infer(
    model = gp_regression(P = P∞, A = A, Q = Q, H = H, var_noise = σ²_noise),
    data = (y = y_data,)
)

λ = sqrt(5)/θ[1];
#### compute matrices for the state-space model ######
L = [0., 0., 1.];
H = [1., 0., 0.];
F = [0. 1. 0.; 0. 0. 1.;-λ^3 -3λ^2 -3λ]
Qc = 16/3 * θ[2] * λ^5;

I = diageye(3) ; 
vec_P = inv(kron(I,F) + kron(F,I)) * vec(-L * Qc * L'); 
P∞ = reshape(vec_P,3,3);
A = [exp(F * i) for i in Δt]; 
Q = [P∞ - i*P∞*i' for i in A];

result_52 = infer(
    model = gp_regression(P = P∞, A = A, Q = Q, H = H, var_noise = σ²_noise),
    data = (y = y_data,)
)

slicedim(dim) = (a) -> map(e -> e[dim], a)

plot(t, mean.(result_32.posteriors[:f]) |> slicedim(1), ribbon = var.(result_32.posteriors[:f]) |> slicedim(1) .|> sqrt, label ="Approx. process_M32", title = "Matern-3/2", legend =false, lw = 2)
plot!(t, mean.(result_52.posteriors[:f]) |> slicedim(1), ribbon = var.(result_52.posteriors[:f]) |> slicedim(1) .|> sqrt, label ="Approx. process_M52",legend = :bottomleft, title = "GPRegression by SSM", lw = 2)
plot!(t, f_true,label="true process", lw = 2)
scatter!(t_obser, f_noisy[pos], label="Observations")
xlabel!("t")
ylabel!("f(t)")