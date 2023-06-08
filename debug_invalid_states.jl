using BSON
include("src/quad_game_utilities.jl")
include("random_polygon_environment.jl")
include("src/plot.jl")
include("fixed_mesh_environment.jl")

data_filename = "output/model-1/best_model.bson"
data = BSON.load(data_filename)[:data]
policy = data["policy"]

wrapper = RandPolyEnv(10:30, 3, "catmull-clark", true)

data_path = "output/debug/"
episodes_per_iteration=50
discount=1.0

rollouts = PPO.Rollouts(data_path)
PPO.collect_rollouts!(rollouts, wrapper, policy, episodes_per_iteration, discount)
