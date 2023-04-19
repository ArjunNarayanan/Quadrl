using BSON
include("../src/quad_game_utilities.jl")
include("../random_polygon_environment.jl")
include("../src/plot.jl")
include("../fixed_mesh_environment.jl")

data_filename = "output/model-1/best_model.bson"
data = BSON.load(data_filename)[:data]
policy = data["policy"]

wrapper = RandPolyEnv(10, 20, "catmull-clark", false)

# PPO.reset!(wrapper)
# original_wrapper = deepcopy(wrapper)
# fig = plot_wrapper(original_wrapper)
# fig.savefig("figures/example_meshes/mesh-5-initial.png")

# best_wrapper = best_state_in_rollout(wrapper, policy)
# fig = plot_wrapper(best_wrapper, mark_geometric_vertices=false)
# fig.savefig("figures/example_meshes/mesh-5-improved.png")

# ret, dev = average_normalized_returns(policy, wrapper, 100)
# ret, dev = average_normalized_best_returns(policy, wrapper, 100)


# mesh, d0 = initialize_random_mesh(10, "catmull-clark")
# wrapper = FixedMeshEnv(mesh, d0, 30)

# plot_wrapper(wrapper)
# ret, dev = average_normalized_returns(policy, wrapper, 100)

# root_dir = "figures/model-1/rollout-3"
# PPO.reset!(wrapper)
# plot_trajectory(policy, wrapper, root_dir)