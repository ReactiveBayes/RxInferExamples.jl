@model function incomplete_data(y, dim)
    Λ ~ Wishart(dim, diagm(ones(dim)))
    m ~ MvNormal(mean=zeros(dim), precision=diagm(ones(dim)))
    for i in 1:size(y, 1)
        x[i] ~ MvNormal(mean=m, precision=Λ)
        for j in 1:dim
            y[i, j] ~ softdot(x[i], StandardBasisVector(dim, j), huge)
        end