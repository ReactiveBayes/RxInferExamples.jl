@model function rx_identification(f, m_x_0, τ_x_0, m_w_0, τ_w_0, a_x, b_x, a_y, b_y, a_w, b_w, y)
    x0 ~ Normal(mean = m_x_0, precision = τ_x_0)
    τ_x ~ Gamma(shape = a_x, rate = b_x)
    w0 ~ Normal(mean = m_w_0, precision = τ_w_0)
    τ_w ~ Gamma(shape = a_w, rate = b_w)
    τ_y ~ Gamma(shape = a_y, rate = b_y)
    
    x ~ Normal(mean = x0, precision = τ_x)
    w ~ Normal(mean = w0, precision = τ_w)

    s := f(x, w)
    y ~ Normal(mean = s, precision = τ_y)
    
end