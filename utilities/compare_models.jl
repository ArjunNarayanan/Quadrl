using BSON
using TOML
include("../src/quad_game_utilities.jl")
include("../random_polygon_environment.jl")
include("../fixed_mesh_environment.jl")
include("../src/plot.jl")

function initialize_random_environment(env_config)
    polygon_degree_list = env_config["min_polygon_degree"]:env_config["max_polygon_degree"]
    env = RandPolyEnv(
        polygon_degree_list,
        env_config["max_actions_factor"],
        env_config["quad_alg"],
        env_config["cleanup"],
        env_config["round_desired_degree"]
    )
    return env
end

function initialize_fixed_environment(env_config)
    min_poly_degree = env_config["min_polygon_degree"]
    max_polyg_degree = env_config["max_polygon_degree"]
    polygon_degree = rand(min_poly_degree:max_polyg_degree)
    max_actions = env_config["max_actions_factor"] * polygon_degree
    mesh, d0 = initialize_random_mesh(
        polygon_degree,
        env_config["quad_alg"],
        env_config["round_desired_degree"]
    )
    cleanup = env_config["cleanup"]
    env = FixedMeshEnv(mesh, d0, max_actions, cleanup)
    return env
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

function load_model(model_name)
    input_dir = joinpath("output", model_name)
    data_filename = joinpath(input_dir, "best_model.bson")
    data = BSON.load(data_filename)[:data]
    policy = data["policy"]
    return policy
end

model_discrete_name = "model-2"
model_discrete = load_model(model_discrete_name)

model_continuous_name = "model-3"
model_continuous = load_model(model_continuous_name)

discrete_config_file = joinpath("output", model_discrete_name, "config.toml")
config = TOML.parsefile(discrete_config_file)
discrete_env_config = config["environment"]

continuous_config_file = joinpath("output", model_continuous_name, "config.toml")
config = TOML.parsefile(continuous_config_file)
wrapper = initialize_fixed_environment(config["environment"])
PPO.reset!(wrapper)

fig = plot_wrapper(wrapper)

# rollout = 1

# output_dir = joinpath(
#     "output",
#     "comparison",
# )
# discrete_output_dir = joinpath(
#     output_dir,
#     "discrete",
#     "rollout-"*string(rollout)
# )
# continuous_output_dir = joinpath(
#     output_dir,
#     "continuous",
#     "rollout-"*string(rollout)
# )
# if !isdir(discrete_output_dir)
#     mkpath(discrete_output_dir)
# end
# if !isdir(continuous_output_dir)
#     mkpath(continuous_output_dir)
# end

# rollout = 1
# PPO.reset!(wrapper)
# plot_trajectory(model_discrete, wrapper, discrete_output_dir)
# PPO.reset!(wrapper)
# plot_trajectory(model_continuous, wrapper, continuous_output_dir)

# rollout += 1