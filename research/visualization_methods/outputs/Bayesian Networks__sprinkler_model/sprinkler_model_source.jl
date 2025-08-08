@model function sprinkler_model(wet_grass)
    clouded ~ Categorical([0.5, 0.5]) # Probability of cloudy being false or true
    rain ~ DiscreteTransition(clouded, [0.8 0.2; 0.2 0.8])
    sprinkler ~ DiscreteTransition(clouded, [0.5 0.9; 0.5 0.1])
    wet_grass ~ DiscreteTransition(sprinkler, [1.0 0.1; 0.0 0.9;;; 0.1 0.01; 0.9 0.99], rain)
end