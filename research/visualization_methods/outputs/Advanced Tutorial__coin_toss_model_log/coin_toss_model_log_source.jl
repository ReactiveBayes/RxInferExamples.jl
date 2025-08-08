@model function coin_toss_model_log(y)
    θ ~ Beta(2.0, 7.0) where { pipeline = LoggerPipelineStage("θ") }
    for i in eachindex(y)
        y[i] ~ Bernoulli(θ)  where { pipeline = LoggerPipelineStage("y[$i]") }
    end