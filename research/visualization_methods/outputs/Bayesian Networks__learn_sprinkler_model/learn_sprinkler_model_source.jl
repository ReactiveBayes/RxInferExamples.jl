@model function learn_sprinkler_model(clouded_data, rain_data, sprinkler_data, wet_grass_data)
    cpt_cloud_rain ~ DirichletCollection(ones(2, 2))
    cpt_cloud_sprinkler ~ DirichletCollection(ones(2, 2))
    cpt_sprinkler_rain_wet_grass ~ DirichletCollection(ones(2, 2, 2))
    for i in 1:length(clouded_data)
        wet_grass_data[i] ~ sprinkler_model(clouded_data = clouded_data[i], rain_data = rain_data[i], sprinkler_data = sprinkler_data[i], cpt_cloud_rain = cpt_cloud_rain, cpt_cloud_sprinkler = cpt_cloud_sprinkler, cpt_sprinkler_rain_wet_grass = cpt_sprinkler_rain_wet_grass)
    end