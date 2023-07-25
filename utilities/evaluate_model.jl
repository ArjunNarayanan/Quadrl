using BSON
using TOML
include("../src/quad_game_utilities.jl")
include("../random_polygon_environment.jl")
include("../fixed_mesh_environment.jl")
include("../src/plot.jl")

function initialize_environment(env_config)
    polygon_degree_list = env_config["min_polygon_degree"]:env_config["max_polygon_degree"]
    env = RandPolyEnv(
        polygon_degree_list,
        env_config["max_actions_factor"],
        env_config["quad_alg"],
        env_config["cleanup"]
    )
    return env
end

function initialize_fixed_environment(env_config)
    min_poly_degree = env_config["min_polygon_degree"]
    max_polyg_degree = env_config["max_polygon_degree"]
    polygon_degree = rand(min_poly_degree:max_polyg_degree)
    max_actions = env_config["max_actions_factor"]*polygon_degree
    mesh, d0 = initialize_random_mesh(
        polygon_degree,
        env_config["quad_alg"]
    )
    cleanup = env_config["cleanup"]
    env = FixedMeshEnv(mesh, d0, max_actions, cleanup)
end

function average_best_fixed_environment_returns(
    policy, 
    num_trajectories, 
    num_samples
)
    ret = zeros(num_samples)
    for sample in 1:num_samples
        wrapper = initialize_fixed_environment(env_config)
        ret[sample] = best_normalized_best_return(policy, wrapper, num_trajectories)
    end
    return Flux.mean(ret), Flux.std(ret)
end

model_name = "model-1"
input_dir = joinpath("output", model_name)
data_filename = joinpath(input_dir, "best_model.bson")
data = BSON.load(data_filename)[:data];
policy = data["policy"]

config_file = joinpath(input_dir, "config.toml")
config = TOML.parsefile(config_file)

env_config = config["environment"]

wrapper = initialize_environment(config["environment"])
PPO.reset!(wrapper)

# fig = plot_wrapper(wrapper)
# fig.savefig("figures/example_meshes/mesh-2.png")
# ret, dev = average_normalized_returns(policy, wrapper, 100)
ret, dev = average_normalized_best_returns(policy, wrapper, 100)
ret, dev = average_best_fixed_environment_returns(
    policy,
    20,
    100
)

# rollout = 1
# output_dir = joinpath(input_dir, "figures", "rollout-"*string(rollout))
# PPO.reset!(wrapper)
# plot_trajectory(policy, wrapper, output_dir)
# rollout += 1