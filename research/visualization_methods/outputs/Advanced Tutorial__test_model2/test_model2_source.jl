@model function test_model2(y)
    
    if length(y) <= 1
        error("The `length` of `y` argument must be greater than one.")
    end