@model function partially_pooled_with_smoking(patient_codes, smoking_status_patient_mapping, weeks, data)
    μ_α_global ~ Normal(mean = 0.0, var = 250000.0) # Prior for the mean of α (intercept)
    μ_β_global ~ Normal(mean = 0.0, var = 250000.0) # Prior for the mean of β (slope)
    σ_α_global ~ Gamma(shape = 1.75, scale = 45.54) # Corresponds to half-normal with scale 100.0
    σ_β_global ~ Gamma(shape = 1.75, scale = 1.36)  # Corresponds to half-normal with scale 3.0

    n_codes = length(patient_codes) # Total number of data points
    n_smoking_statuses = length(unique(smoking_status_patient_mapping)) # Number of different smoking patterns
    n_patients = length(unique(patient_codes)) # Number of unique patients in the data

    local μ_α_smoking_status  # Individual intercepts for smoking pattern
    local μ_β_smoking_status  # Individual slopes for smoking pattern
    
    for i in 1:n_smoking_statuses
        μ_α_smoking_status[i] ~ Normal(mean = μ_α_global, precision = σ_α_global)
        μ_β_smoking_status[i] ~ Normal(mean = μ_β_global, precision = σ_β_global)
    end