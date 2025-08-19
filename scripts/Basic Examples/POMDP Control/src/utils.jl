using Distributions

function grid_location_to_index(pos::Tuple{Int, Int})
    return (pos[2] - 1) * 5 + pos[1]
end

function index_to_grid_location(index::Int)
    return (index % 5, index ÷ 5 + 1,)
end

function index_to_one_hot(index::Int)
    return [i == index ? 1.0 : 0.0 for i in 1:25]
end


