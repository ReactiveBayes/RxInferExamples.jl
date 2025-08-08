@model function lotka_volterra_model(obs, mprev, Vprev, dt, t, mθ, Vθ)
    θ     ~ MvNormalMeanCovariance(mθ, Vθ)
    xprev ~ MvNormalMeanCovariance(mprev, Vprev)
    x     := lotka_volterra_rk4(xprev, θ, t, dt)
    obs   ~ MvNormalMeanCovariance(x,  noisev * diageye(length(mprev)))
end