include("../src/quad_game_utilities.jl")
include("../random_polygon_environment.jl")
using BenchmarkTools

function evaluate_cpu(policy, state)
    return PPO.action_probabilities(policy, state)
end

wrapper = RandPolyEnv(20, 20, "catmull-clark")
policy = SimplePolicy.Policy(216, 128, 5, 5)
gpu_policy = policy |> gpu

rollouts = PPO.EpisodeData()
PPO.collect_rollouts!(rollouts, wrapper, gpu_policy, 100)