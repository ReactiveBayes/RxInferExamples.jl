@model function linear_regression_multivariate(dim, x, y)
    a ~ MvNormal(mean = zeros(dim), covariance = 100 * diageye(dim))
    b ~ MvNormal(mean = ones(dim), covariance = 100 * diageye(dim))
    W ~ InverseWishart(dim + 2, 100 * diageye(dim))
    y .~ MvNormal(mean = x .* a .+ b, covariance = W)
end