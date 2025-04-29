# This file was automatically generated from /home/trim/Documents/GitHub/RxInferExamples.jl/examples/Basic Examples/Bayesian Networks/Bayesian Networks.ipynb
# by notebooks_to_scripts.jl at 2025-04-29T06:39:07.328
#
# Source notebook: Bayesian Networks.ipynb

using RxInfer, Plots, GraphViz

@model function sprinkler_model(wet_grass)
    clouded ~ Categorical([0.5, 0.5]) # Probability of cloudy being false or true
    rain ~ DiscreteTransition(clouded, [0.8 0.2; 0.2 0.8])
    sprinkler ~ DiscreteTransition(clouded, [0.5 0.9; 0.5 0.1])
    wet_grass ~ DiscreteTransition(sprinkler, [1.0 0.1; 0.0 0.9;;; 0.1 0.01; 0.9 0.99], rain)
end

model_generator = sprinkler_model() | (wet_grass = [ 1.0, 0.0 ], )
model_to_plot   = RxInfer.getmodel(RxInfer.create_model(model_generator))
GraphViz.load(model_to_plot, strategy = :simple)

initialization = @initialization begin
    μ(sprinkler) = Categorical([0.5, 0.5])
end

data = (wet_grass = [1.0, 0.0],) # Grass is dry

result = infer(model=sprinkler_model(), data=data, iterations=10, initialization=initialization)

p1 = bar(last(result.posteriors[:clouded]).p,
    xticks=(1:2, ["Not Clouded", "Clouded"]),
    ylabel="Probability",
    title="Posterior Probability of Clouded Variable",
    titlefontsize=10,
    legend=false)

p2 = bar(last(result.posteriors[:rain]).p,
    xticks=(1:2, ["No Rain", "Rain"]),
    ylabel="Probability", 
    title="Posterior Probability of Rain Variable",
    titlefontsize=10,
    legend=false)

p3 = bar(last(result.posteriors[:sprinkler]).p,
    xticks=(1:2, ["Off", "On"]),
    ylabel="Probability",
    title="Posterior Probability of Sprinkler Variable", 
    titlefontsize=10,
    legend=false)

plot(p1, p2, p3, layout=(1,3), size=(900,300))


result = infer(model=sprinkler_model(), data=(wet_grass=[0.0, 1.0],), iterations=10, initialization=initialization)
p1 = bar(last(result.posteriors[:clouded]).p,
    xticks=(1:2, ["Not Clouded", "Clouded"]),
    ylabel="Probability",
    title="Posterior Probability of Clouded Variable",
    titlefontsize=10,
    legend=false)

p2 = bar(last(result.posteriors[:rain]).p,
    xticks=(1:2, ["No Rain", "Rain"]),
    ylabel="Probability", 
    title="Posterior Probability of Rain Variable",
    titlefontsize=10,
    legend=false)

p3 = bar(last(result.posteriors[:sprinkler]).p,
    xticks=(1:2, ["Off", "On"]),
    ylabel="Probability",
    title="Posterior Probability of Sprinkler Variable", 
    titlefontsize=10,
    legend=false)

plot(p1, p2, p3, layout=(1,3), size=(900,300))

@model function sprinkler_model(wet_grass_data, sprinkler_data, rain_data, clouded_data)
    clouded ~ Categorical([0.5, 0.5]) # Probability of cloudy being false or true
    clouded_data ~ DiscreteTransition(clouded, diageye(2))  
    rain ~ DiscreteTransition(clouded, [0.8 0.2; 0.2 0.8])
    rain_data ~ DiscreteTransition(rain, diageye(2))
    sprinkler ~ DiscreteTransition(clouded, [0.5 0.9; 0.5 0.1])
    sprinkler_data ~ DiscreteTransition(sprinkler, diageye(2))
    wet_grass ~ DiscreteTransition(sprinkler, [1.0 0.1; 0.0 0.9;;; 0.1 0.01; 0.9 0.99], rain)
    wet_grass_data ~ DiscreteTransition(wet_grass, diageye(2))
end

result = infer(model=sprinkler_model(), data=(wet_grass_data=[0.0, 1.0], sprinkler_data=[0.0, 1.0], rain_data=missing, clouded_data=missing), iterations=10, initialization=initialization)


result = infer(model=sprinkler_model(), data=(wet_grass_data=missing, sprinkler_data=[1.0, 0.0], rain_data=[0.0, 1.0], clouded_data=missing), iterations=10, initialization=initialization)


p1 = bar(last(result.posteriors[:rain]).p,
    xticks=(1:2, ["No", "Yes"]),
    ylabel="Probability", 
    title="Posterior Probability of Rain Variable",
    titlefontsize=8,
    legend=false)

p2 = bar(last(result.posteriors[:clouded]).p,
    xticks=(1:2, ["No", "Yes"]),
    ylabel="Probability",
    title="Posterior Probability of Clouded Variable",
    titlefontsize=8,
    legend=false)

p3 = bar(last(result.posteriors[:sprinkler]).p,
    xticks=(1:2, ["Off", "On"]),
    ylabel="Probability",
    title="Posterior Probability of Sprinkler Variable", 
    titlefontsize=8,
    legend=false)

p4 = bar(last(result.posteriors[:wet_grass]).p,
    xticks=(1:2, ["No", "Yes"]),
    ylabel="Probability",
    title="Posterior Probability of Wet Grass Variable",
    titlefontsize=8,
    legend=false)

plot(p1, p2, p3, p4, layout=(1,4), size=(1200,300))



result = infer(model=sprinkler_model(), data=(wet_grass_data=[0.0, 1.0], sprinkler_data=missing, rain_data=missing, clouded_data=[1.0, 0.0]), iterations=10, initialization=initialization)


p1 = bar(last(result.posteriors[:rain]).p,
    xticks=(1:2, ["No", "Yes"]),
    ylabel="Probability", 
    title="Posterior Probability of Rain Variable",
    titlefontsize=8,
    legend=false)

p2 = bar(last(result.posteriors[:clouded]).p,
    xticks=(1:2, ["No", "Yes"]),
    ylabel="Probability",
    title="Posterior Probability of Clouded Variable",
    titlefontsize=8,
    legend=false)

p3 = bar(last(result.posteriors[:sprinkler]).p,
    xticks=(1:2, ["Off", "On"]),
    ylabel="Probability",
    title="Posterior Probability of Sprinkler Variable", 
    titlefontsize=8,
    legend=false)

p4 = bar(last(result.posteriors[:wet_grass]).p,
    xticks=(1:2, ["No", "Yes"]),
    ylabel="Probability",
    title="Posterior Probability of Wet Grass Variable",
    titlefontsize=8,
    legend=false)

plot(p1, p2, p3, p4, layout=(1,4), size=(1200,300))



# Generate synthetic data from the true model
n_samples = 10000

# Initialize arrays to store the samples
clouded_samples = zeros(Int, n_samples)
rain_samples = zeros(Int, n_samples)
sprinkler_samples = zeros(Int, n_samples) 
wet_grass_samples = zeros(Int, n_samples)

# Sample from the model
for i in 1:n_samples
    # Sample clouded (prior)
    clouded_samples[i] = rand() < 0.5 ? 1 : 2
    
    # Sample rain (depends on clouded)
    rain_prob = clouded_samples[i] == 1 ? 0.2 : 0.8
    rain_samples[i] = rand() > rain_prob ? 1 : 2
    
    # Sample sprinkler (depends on clouded)
    sprinkler_prob = clouded_samples[i] == 1 ? 0.5 : 0.1
    sprinkler_samples[i] = rand() > sprinkler_prob ? 1 : 2
    
    # Sample wet grass (depends on rain and sprinkler)
    if rain_samples[i] == 2 && sprinkler_samples[i] == 2
        wet_prob = 0.99
    elseif rain_samples[i] == 2
        wet_prob = 0.9
    elseif sprinkler_samples[i] == 2
        wet_prob = 0.9
    else
        wet_prob = 0.0
    end
    wet_grass_samples[i] = rand() < wet_prob ? 2 : 1
end
# Convert to one-hot encoding
clouded_data = [[i == s ? 1.0 : 0.0 for i in 1:2] for s in clouded_samples]
rain_data = [[i == s ? 1.0 : 0.0 for i in 1:2] for s in rain_samples]
sprinkler_data = [[i == s ? 1.0 : 0.0 for i in 1:2] for s in sprinkler_samples]
wet_grass_data = [[i == s ? 1.0 : 0.0 for i in 1:2] for s in wet_grass_samples];

@model function sprinkler_model(clouded_data, rain_data, sprinkler_data, wet_grass_data, cpt_cloud_rain, cpt_cloud_sprinkler, cpt_sprinkler_rain_wet_grass)
    clouded ~ Categorical([0.5, 0.5]) # Probability of cloudy being false or true
    clouded_data ~ DiscreteTransition(clouded, diageye(2))
    rain ~ DiscreteTransition(clouded, cpt_cloud_rain)
    rain_data ~ DiscreteTransition(rain, diageye(2))
    sprinkler ~ DiscreteTransition(clouded, cpt_cloud_sprinkler)
    sprinkler_data ~ DiscreteTransition(sprinkler, diageye(2))
    wet_grass ~ DiscreteTransition(sprinkler, cpt_sprinkler_rain_wet_grass, rain)
    wet_grass_data ~ DiscreteTransition(wet_grass, diageye(2))
end

@model function learn_sprinkler_model(clouded_data, rain_data, sprinkler_data, wet_grass_data)
    cpt_cloud_rain ~ DirichletCollection(ones(2, 2))
    cpt_cloud_sprinkler ~ DirichletCollection(ones(2, 2))
    cpt_sprinkler_rain_wet_grass ~ DirichletCollection(ones(2, 2, 2))
    for i in 1:length(clouded_data)
        wet_grass_data[i] ~ sprinkler_model(clouded_data = clouded_data[i], rain_data = rain_data[i], sprinkler_data = sprinkler_data[i], cpt_cloud_rain = cpt_cloud_rain, cpt_cloud_sprinkler = cpt_cloud_sprinkler, cpt_sprinkler_rain_wet_grass = cpt_sprinkler_rain_wet_grass)
    end
end


initialization = @initialization begin
    q(cpt_cloud_rain) = DirichletCollection(ones(2, 2))
    q(cpt_cloud_sprinkler) = DirichletCollection(ones(2, 2))
    q(cpt_sprinkler_rain_wet_grass) = DirichletCollection(ones(2, 2, 2))
    for init in sprinkler_model
        μ(sprinkler) = Categorical([0.5, 0.5])
    end
end

constraints = @constraints begin
    for q in sprinkler_model
        q(cpt_cloud_rain, clouded, rain) = q(clouded,rain)q(cpt_cloud_rain)
        q(cpt_cloud_sprinkler, clouded, sprinkler) = q(clouded,sprinkler)q(cpt_cloud_sprinkler)
        q(cpt_sprinkler_rain_wet_grass, sprinkler, rain, wet_grass) = q(sprinkler,rain,wet_grass)q(cpt_sprinkler_rain_wet_grass)
    end
end

result = infer(model=learn_sprinkler_model(), 
            data=(clouded_data=clouded_data, rain_data=rain_data, sprinkler_data=sprinkler_data, wet_grass_data=wet_grass_data), 
            constraints=constraints, 
            initialization=initialization, 
            iterations=5, 
            showprogress=true,
            options=(limit_stack_depth=500,))

using Plots

# Plot CPT for cloud -> rain
cloud_rain = mean(last(result.posteriors[:cpt_cloud_rain]))
p1 = heatmap(cloud_rain, 
        title="P(Rain | Cloudy)", 
        xlabel="Cloudy", 
        ylabel="Rain",
        xticks=(1:2, ["False", "True"]),
        yticks=(1:2, ["False", "True"]),
        xrotation=45,
        left_margin=10Plots.mm,
        bottom_margin=10Plots.mm)

# Plot CPT for cloud -> sprinkler
cloud_sprinkler = mean(last(result.posteriors[:cpt_cloud_sprinkler]))
p2 = heatmap(cloud_sprinkler,
        title="P(Sprinkler | Cloudy)",
        xlabel="Cloudy",
        ylabel="Sprinkler", # Remove y-label since it's shown in p1
        xticks=(1:2, ["False", "True"]),
        yticks=(1:2, ["False", "True"]),
        xrotation=45,
        bottom_margin=10Plots.mm)

# Plot CPT for sprinkler,rain -> wet grass
sprinkler_rain_wet = mean(last(result.posteriors[:cpt_sprinkler_rain_wet_grass]))
p3 = heatmap(sprinkler_rain_wet[:,:,1],
        title="P(Wet Grass=False | Sprinkler,Rain)",
        xlabel="Rain",
        ylabel="Sprinkler", # Remove y-label since it's shown in p1
        xticks=(1:2, ["False", "True"]),
        yticks=(1:2, ["False", "True"]),
        xrotation=45,
        bottom_margin=10Plots.mm)
p4 = heatmap(sprinkler_rain_wet[:,:,2],
        title="P(Wet Grass=True | Sprinkler,Rain)", 
        xlabel="Rain",
        ylabel="Sprinkler", # Remove y-label since it's shown in p1
        xticks=(1:2, ["False", "True"]),
        yticks=(1:2, ["False", "True"]),
        xrotation=45,
        bottom_margin=15Plots.mm)

plot(p1, p2, p3, p4, layout=(1,4), size=(1700,305))