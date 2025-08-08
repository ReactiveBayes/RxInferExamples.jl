@model function dot_sequence(out, k, i, orders, x, c, b)
    if k === length(orders)
        out ~ b[k] * dot(c[k], x[k, i])
    else 
        next ~ dot_sequence(k = k + 1, i = i, orders = orders, x = x, c = c, b = b)
        out  ~ b[k] * dot(c[k], x[k, i]) + next
    end