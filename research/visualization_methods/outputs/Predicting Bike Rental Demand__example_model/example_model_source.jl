@model function example_model(y)

    h ~ NormalMeanPrecision(0, 1.0)
    x ~ NormalMeanPrecision(h, 1.0)
    y ~ NormalMeanPrecision(x, 10.0)
end