using BSON
include("../src/quad_game_utilities.jl")
include("../random_polygon_environment.jl")


data_filename = "output/model-1/best_model.bson"
data = BSON.load(data_filename)[:data]
policy = data["policy"]

env = RandPolyEnv(10, 20, "catmull-clark")
# ret, dev = average_normalized_returns(policy, env, 100)
# ret, dev = average_normalized_best_returns(policy, env, 100)

include("../src/plot.jl")
root_dir = "output/model-1/figures/rollout-2"
PPO.reset!(env)
plot_trajectory(policy, env, root_dir)