using BSON
include("../src/quad_game_utilities.jl")
include("../random_polygon_environment.jl")
include("../src/plot.jl")
include("../fixed_mesh_environment.jl")

data_filename = "output/model-1/best_model.bson"
data = BSON.load(data_filename)[:data]
policy = data["policy"]

wrapper = RandPolyEnv(10, 20, "catmull-clark")

PPO.reset!(wrapper)
original_wrapper = deepcopy(wrapper)
fig = plot_wrapper(original_wrapper)
# fig.savefig("figures/example_meshes/mesh-4.png")

# best_wrapper = best_state_in_rollout(wrapper, policy)
# QM.cleanup_env!(best_wrapper.env, 20)
# fig = plot_wrapper(best_wrapper, mark_geometric_vertices=true)
# fig.savefig("figures/example_meshes/mesh-4-improved.png")

# state = PPO.state(wrapper)
# action_probabilities = PPO.action_probabilities(policy, state)
# action = rand(Categorical(action_probabilities))
# PPO.step!(wrapper, action)

# fig = plot_wrapper(wrapper)
# fig.savefig("figures/example_meshes/mesh-3.png")

# ret, dev = average_normalized_returns(policy, wrapper, 100)
# ret, dev = average_normalized_best_returns(policy, env, 100)


# mesh, d0 = initialize_random_mesh(10, "catmull-clark")
# wrapper = FixedMeshEnv(mesh, d0, 30)

# plot_wrapper(wrapper)
# ret, dev = average_normalized_returns(policy, wrapper, 100)

# root_dir = "output/model-1/figures/rollout-2"
# PPO.reset!(env)
# plot_trajectory(policy, env, root_dir)