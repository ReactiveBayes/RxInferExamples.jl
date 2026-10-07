using Reactant, LinearAlgebra, RxInfer
const FC = RxInfer.FastCholesky
A = [4.0 1.0 0.5; 1.0 3.0 0.2; 0.5 0.2 2.0]; B = [1.0 2.0 3.0; 0.0 1.0 -1.0]; b = [1.0, 2.0, 3.0]
rA, rB, rb = Reactant.to_rarray.((A, B, b))
tests = [
  ("fastcholesky(A).U",  (A, B, b) -> FC.fastcholesky(A).U * A, cholesky(A).U * A),
  ("fastcholesky(A).L",  (A, B, b) -> FC.fastcholesky(A).L * A, cholesky(A).L * A),
  ("C \\ b",             (A, B, b) -> FC.fastcholesky(A) \ b, A \ b),
  ("B / C",              (A, B, b) -> B / FC.fastcholesky(A), B / A),
  ("Bt' adjoint / C",    (A, B, b) -> permutedims(B)' / FC.fastcholesky(A), permutedims(B)' / A),
  ("cholinv(A)",         (A, B, b) -> FC.cholinv(A), inv(A)),
  ("chollogdet(A)",      (A, B, b) -> FC.chollogdet(A), logdet(A)),
  ("cholinv_logdet(A)",  (A, B, b) -> (r = FC.cholinv_logdet(A); r[1] * r[2]), inv(A) * logdet(A)),
  ("cholsqrt(A)",        (A, B, b) -> FC.cholsqrt(A) * A, cholesky(A).L * A),
]
for (name, f, ref) in tests
  r = try
    x = @jit f(rA, rB, rb); v = x isa Number ? x : Array(x)
    "ok   max|Δ| = " * string(maximum(abs, v .- ref))
  catch e
    "FAIL " * first(split(sprint(showerror, e), "\n"))
  end
  println(rpad(name, 22), r)
end
