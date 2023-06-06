using BSON
include("src/quad_game_utilities.jl")
include("random_polygon_environment.jl")
include("src/plot.jl")
include("fixed_mesh_environment.jl")

data_filename = "output/model-1/best_model.bson"
data = BSON.load(data_filename)[:data]
policy = data["policy"]

wrapper = RandPolyEnv(10:30, 3, "catmull-clark", true)
ret, dev = average_normalized_returns(policy, wrapper, 10000)
