# This file was automatically generated from /home/trim/Documents/GitHub/RxInferExamples.jl/examples/Problem Specific/Litter Model/Litter Model.ipynb
# by notebooks_to_scripts.jl at 2025-08-07T12:32:29.003
#
# Source notebook: Litter Model.ipynb

using RxInfer, Random, Distributions, Plots, LaTeXStrings, XLSX, DataFrames

## provision function, provides another state/datapoint from simulation
function fˢⁱᵐₚ(s; θ̆, 𝙼, 𝚅, 𝙲, rng)
    dp = Vector{Vector{Vector{Float64}}}(undef, 𝙼)
    for m in 1:𝙼 ## Matrices
        dp[m] = Vector{Vector{Float64}}(undef, 𝚅)
        for v in 1:𝚅 ## Vectors
            dp[m][v] = Vector{Float64}(undef, 𝙲)
            for c in 1:𝙲 ## Components
               dp[m][v][c] = float(rand(rng, Poisson(θ̆)))
            end
        end
    end
    s̆ = dp
    return s̆
end

_s = 1 ## s for sequence
_θ̆ˢⁱᵐ = 15 ## lambda of Poisson distribution
_rng = MersenneTwister(57)
## _s̆ = fˢⁱᵐₚ(_s, θ̆=_θ̆ˢⁱᵐ, 𝙼=3, 𝚅=4, 𝙲=5, rng=_rng) ## color image with 3 colors, 4 rows, 5 cols of elements
## _s̆ = fˢⁱᵐₚ(_s, θ̆=_θ̆ˢⁱᵐ, 𝙼=1, 𝚅=4, 𝙲=5, rng=_rng) ## b/w image with 4 rows, 5 cols of elements
_s̆ = fˢⁱᵐₚ(_s, θ̆=_θ̆ˢⁱᵐ, 𝙼=1, 𝚅=1, 𝙲=5, rng=_rng) ## vector with 5 elements
## _s̆ = fˢⁱᵐₚ(_s, θ̆=_θ̆ˢⁱᵐ, 𝙼=1, 𝚅=1, 𝙲=1, rng=_rng) ## vector with 1 element
;

## provision function, provides another state/datapoint from field
function fᶠˡᵈₚ(s; 𝙼, 𝚅, 𝙲, df)
    dp = Vector{Vector{Vector{Float64}}}(undef, 𝙼)
    for m in 1:𝙼 ## Matrices
        dp[m] = Vector{Vector{Float64}}(undef, 𝚅)
        for v in 1:𝚅 ## Vectors
            dp[m][v] = Vector{Float64}(undef, 𝙲)
            for c in 1:𝙲 ## Components
                dp[m][v][c] = df[s, :incidents]
            end
        end
    end
    s̆ = dp
    return s̆
end
## _s = 1 ## s for sequence
## dp = fᶠˡᵈₚ(_s, 𝙼=3, 𝚅=4, 𝙲=5, df=_fld_df) ## color image with 3 colors, 4 rows, 5 cols of elements
## dp = fᶠˡᵈₚ(_s, 𝙼=1, 𝚅=4, 𝙲=5, df=_fld_df) ## b/w image with 4 rows, 5 cols of elements
## dp = fᶠˡᵈₚ(_s, 𝙼=1, 𝚅=1, 𝙲=5, df=_fld_df) ## vector with 5 elements
## dp = fᶠˡᵈₚ(_s, 𝙼=1, 𝚅=1, 𝙲=1, df=_fld_df) ## vector with 1 element

## response function, provides the response to a state/datapoint
function fᵣ(s̆)
    return s̆ ## no noise
end
fᵣ(_s̆);

## Data comes from either a simulation/lab (sim|lab) OR from the field (fld)
## Data are handled either in batches (batch) OR online as individual points (point)
function sim_data(rng, 𝚂, 𝙳, 𝙼, 𝚅, 𝙲, θ̆)
    p = Vector{Vector{Vector{Vector{Vector{Float64}}}}}(undef, 𝚂)
    s̆ = Vector{Vector{Vector{Vector{Vector{Float64}}}}}(undef, 𝚂)
    r = Vector{Vector{Vector{Vector{Vector{Float64}}}}}(undef, 𝚂)
    y = Vector{Vector{Vector{Vector{Vector{Float64}}}}}(undef, 𝚂)
    for s in 1:𝚂 ## sequences
        p[s] = Vector{Vector{Vector{Vector{Float64}}}}(undef, 𝙳)
        s̆[s] = Vector{Vector{Vector{Vector{Float64}}}}(undef, 𝙳)
        r[s] = Vector{Vector{Vector{Vector{Float64}}}}(undef, 𝙳)
        y[s] = Vector{Vector{Vector{Vector{Float64}}}}(undef, 𝙳)
        for d in 1:𝙳 ## datapoints
            p[s][d] = fˢⁱᵐₚ(s; θ̆=θ̆, 𝙼=𝙼, 𝚅=𝚅, 𝙲=𝙲, rng=rng)
            s̆[s][d] = p[s][d] ## no system noise
            r[s][d] = fᵣ(s̆[s][d])
            y[s][d] = r[s][d]
        end
    end
    return y
end;

function fld_data(df, 𝚂, 𝙳, 𝙼, 𝚅, 𝙲)
    p = Vector{Vector{Vector{Vector{Vector{Float64}}}}}(undef, 𝚂)
    s̆ = Vector{Vector{Vector{Vector{Vector{Float64}}}}}(undef, 𝚂)
    r = Vector{Vector{Vector{Vector{Vector{Float64}}}}}(undef, 𝚂)
    y = Vector{Vector{Vector{Vector{Vector{Float64}}}}}(undef, 𝚂)
    for s in 1:𝚂 ## sequences
        p[s] = Vector{Vector{Vector{Vector{Float64}}}}(undef, 𝙳)
        s̆[s] = Vector{Vector{Vector{Vector{Float64}}}}(undef, 𝙳)
        r[s] = Vector{Vector{Vector{Vector{Float64}}}}(undef, 𝙳)
        y[s] = Vector{Vector{Vector{Vector{Float64}}}}(undef, 𝙳)
        for d in 1:𝙳 ## datapoints
            p[s][d] = fᶠˡᵈₚ(s; 𝙼=𝙼, 𝚅=𝚅, 𝙲=𝙲, df=df)
            s̆[s][d] = p[s][d] ## no system noise
            r[s][d] = fᵣ(s̆[s][d])
            y[s][d] = r[s][d]
        end
    end
    return y
end;

## number of Batches in an experiment
## _𝙱 = 1 ## not used yet

## number of Sequences/examples in a batch
_𝚂 = 365
## _𝚂 = 3

## number of Datapoints in a sequence
_𝙳 = 1
## _𝙳 = 2
## _𝙳 = 3

## number of Matrices in a datapoint
_𝙼 = 1

## number of Vectors in a matrix
_𝚅 = 1

## number of Components in a vector
_𝙲 = 1

_θ̆ˢⁱᵐ = 15 ## hidden lambda of Poisson distribution
_rng = MersenneTwister(57);

_yˢⁱᵐ = sim_data(_rng, _𝚂, _𝙳, _𝙼, _𝚅, _𝙲, _θ̆ˢⁱᵐ) ## simulated data
_yˢⁱᵐ = first.(first.(first.(first.(_yˢⁱᵐ))));

## methods(print)
## print(_yˢⁱᵐ[1:2])

## Customize the display width to control positioning or prevent wrapping
## io = IOContext(stdout, :displaysize => (50, 40)) ## (rows, cols)
## print(io, _yˢⁱᵐ[1:3])
## print(io, _yˢⁱᵐ)

print(IOContext(stdout, :displaysize => (24, 5)), _yˢⁱᵐ[1:10]);

_rθ = range(0, _𝚂, length=1*_𝚂)
_p = plot(title="Simulated Daily Litter Events", xlabel="Day")
_p = plot!(_rθ, _yˢⁱᵐ, linetype=:steppre, label="# daily events", c=1)
plot(_p)

## parameters for the prior distribution
_αᴳᵃᵐ, _θᴳᵃᵐ = 350., .05;

## Litter model: Gamma-Poisson
@model function litter_model(x, αᴳᵃᵐ, θᴳᵃᵐ)
    ## prior on θ parameter of the model
    θ ~ Gamma(shape=αᴳᵃᵐ, rate=θᴳᵃᵐ) ## 1 Gamma factor

    ## assume daily number of litter incidents is a Poisson distribution
    for i in eachindex(x)
        x[i] ~ Poisson(θ) ## not θ̃; N Poisson factors
    end
end

_result = infer(
    model= litter_model(αᴳᵃᵐ= _αᴳᵃᵐ, θᴳᵃᵐ= _θᴳᵃᵐ), 
    data= (x= _yˢⁱᵐ, )
)

_θˢⁱᵐ = _result.posteriors[:θ]

_rθ = range(0, 20, length=500)
_p = plot(title="Simulation results: Distribution of "*L"θ^{\mathrm{sim}}=λ")
plot!(_rθ, (x) -> pdf(Gamma(_αᴳᵃᵐ, _θᴳᵃᵐ), x), fillalpha=0.3, fillrange=0, label="P(θ)", c=1,)
plot!(_rθ, (x) -> pdf(_θˢⁱᵐ, x), fillalpha=0.3, fillrange=0, label="P(θ|x)", c=3)
vline!([_θ̆ˢⁱᵐ], label="Hidden θ", c=2)

_fld_df = DataFrame(XLSX.readtable("litter_incidents.xlsx", "Sheet1"))
_yᶠˡᵈ = fld_data(_fld_df, _𝚂, _𝙳, _𝙼, _𝚅, _𝙲) ## field data
_yᶠˡᵈ = first.(first.(first.(first.(_yᶠˡᵈ))))
print(IOContext(stdout, :displaysize => (24, 30)), _yᶠˡᵈ[1:10]);

_rθ = range(0, _𝚂, length=1*_𝚂)
_p = plot(title="Field Daily Litter Events", xlabel="Day")
_p = plot!(_rθ, _yᶠˡᵈ, linetype=:steppre, label="# daily events", c=1)
plot(_p)

_result = infer(
    model=litter_model(αᴳᵃᵐ= _αᴳᵃᵐ, θᴳᵃᵐ= _θᴳᵃᵐ), 
    data= (x= _yᶠˡᵈ, )
)

_θᶠˡᵈ = _result.posteriors[:θ]

_rθ = range(0, 20, length=500)
_p = plot(title="Field results: Distribution of "*L"θ^{\mathrm{fld}}=λ")
plot!(_rθ, (x) -> pdf(Gamma(_αᴳᵃᵐ, _θᴳᵃᵐ), x), fillalpha=0.3, fillrange=0, label="P(θ)", c=1,)
plot!(_rθ, (x) -> pdf(_θᶠˡᵈ, x), fillalpha=0.3, fillrange=0, label="P(θ|x)", c=3)