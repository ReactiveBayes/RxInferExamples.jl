@model function lotka_volterra_model_without_prior(obs, mprev, Vprev, dt, t, θ)
    xprev ~ MvNormalMeanCovariance(mprev, Vprev)
    x     := lotka_volterra_rk4(xprev, θ, t, dt)
    obs   ~ MvNormalMeanCovariance(x,  noisev * diageye(length(mprev)))
end