@model function LVAR(y, orders)

    priors   = form_priors(orders)
    c, b     = form_c_b(y, orders)
    y_length = length(y)
    
    local x # `x` is being initialized in the loop within submodels
    for k in 1:length(orders)
        x ~ AR_sequence(index  = k, length = y_length, priors = priors, order  = orders[k])
    end