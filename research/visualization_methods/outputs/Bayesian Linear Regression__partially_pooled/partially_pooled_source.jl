@model function partially_pooled(patient_codes, weeks, data)
    μ_α ~ Normal(mean = 0.0, var = 250000.0) # Prior for the mean of α (intercept)
    μ_β ~ Normal(mean = 0.0, var = 9.0)      # Prior for the mean of β (slope)
    σ_α ~ Gamma(shape = 1.75, scale = 45.54) # Prior for the precision of α (intercept)
    σ_β ~ Gamma(shape = 1.75, scale = 1.36)  # Prior for the precision of β (slope)

    n_codes = length(patient_codes)            # Total number of data points
    n_patients = length(unique(patient_codes)) # Number of unique patients in the data

    local α # Individual intercepts for each patient
    local β # Individual slopes for each patient

    for i in 1:n_patients
        α[i] ~ Normal(mean = μ_α, precision = σ_α) # Sample the intercept α from a Normal distribution
        β[i] ~ Normal(mean = μ_β, precision = σ_β) # Sample the slope β from a Normal distribution
    end