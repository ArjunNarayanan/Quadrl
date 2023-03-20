using Flux
using PlotQuadMesh
using RandomQuadMesh
using QuadMeshGame
using ProximalPolicyOptimization
using Distributions: Categorical
using BSON
using Printf

RQ = RandomQuadMesh
QM = QuadMeshGame
PPO = ProximalPolicyOptimization
PQ = PlotQuadMesh

const NUM_ACTIONS_PER_EDGE = 5
const NO_ACTION_REWARD = -1

include("state.jl")
include("policy.jl")
include("action_probabilities.jl")
include("step.jl")




#####################################################################################################################
# EVALUATING PERFORMANCE
mutable struct SaveBestModel
    root_dir
    file_path
    num_trajectories
    best_return
    mean_returns
    std_returns
    action_counts
    function SaveBestModel(root_dir, num_trajectories, filename = "best_model.bson")
        if !isdir(root_dir)
            mkpath(root_dir)
        end

        file_path = joinpath(root_dir, filename)
        mean_returns = []
        std_returns = []
        action_counts = []
        new(root_dir, file_path, num_trajectories, -Inf, mean_returns, std_returns, action_counts)
    end
end

function save_model(s::SaveBestModel, policy)
    cpu_policy = cpu(policy)
    data = Dict("policy" => cpu_policy)
    println("SAVING MODEL AT : " * s.file_path * "\n\n")
    BSON.@save s.file_path data
end

function save_evaluator(s::SaveBestModel)
    data = Dict("evaluator" => s)
    filepath = joinpath(s.root_dir, "evaluator.bson")
    println("SAVING EVALUATOR AT : ", filepath, "\n")
    BSON.@save filepath data
end

function (s::SaveBestModel)(policy, wrapper)
    ret, dev, action_counts = average_normalized_returns_and_action_stats(policy, wrapper, s.num_trajectories)
    if ret > s.best_return
        s.best_return = ret
        @printf "\nNEW BEST RETURN : %1.4f\n" ret
        save_model(s, policy)
    end

    @printf "RET = %1.4f\tDEV = %1.4f\n" ret dev
    println("ACTION COUNTS: ", action_counts, "\n")
    push!(s.mean_returns, ret)
    push!(s.std_returns, dev)
    push!(s.action_counts, action_counts)

    save_evaluator(s)

    efficiency = average_mesh_efficiency(policy, wrapper, s.num_trajectories)
    println("AVG MESH EFFICIENCY : ", efficiency)
end

function PPO.save_loss(s::SaveBestModel, loss)
    outfile = joinpath(s.root_dir, "loss.bson")
    BSON.@save outfile loss
end

function single_trajectory_mesh_efficiency(policy, env)
    efficiency_history = Float64[]
    done = PPO.is_terminal(env)

    while !done
        num_quads = QM.number_of_quads(env.env.mesh)
        quad_buffer = QM.quad_buffer(env.env.mesh)
        efficiency = num_quads/quad_buffer
        push!(efficiency_history, efficiency)

        state = PPO.state(env) |> gpu
        probs = PPO.action_probabilities(policy, state) |> cpu
        action = rand(Categorical(probs))
        @assert probs[action] > 0.0
        PPO.step!(env, action)

        done = PPO.is_terminal(env)
    end
    return efficiency_history
end

function average_mesh_efficiency(policy, env, num_trajectories)
    efficiency = Float64[]
    for iter in 1:num_trajectories
        _efficiency = single_trajectory_mesh_efficiency(policy, env)
        append!(efficiency, _efficiency)
    end
    return Flux.mean(efficiency)
end

function single_trajectory_return_and_action_stats(policy, env)
    ret = 0
    action_type_list = Int[]
    done = PPO.is_terminal(env)

    while !done
        state = PPO.state(env) |> gpu
        probs = PPO.action_probabilities(policy, state) |> cpu
        action = rand(Categorical(probs))
        @assert probs[action] > 0.0
        _, _, action_type = index_to_action(action)

        PPO.step!(env, action)

        done = PPO.is_terminal(env)
        ret += PPO.reward(env)
        push!(action_type_list, action_type)
    end
    return ret, action_type_list
end

function single_trajectory_normalized_return_and_action_stats(policy, wrapper)
    maxreturn = wrapper.current_score - wrapper.opt_score
    if maxreturn == 0
        return 1.0, Int[]
    else
        ret, stats = single_trajectory_return_and_action_stats(policy, wrapper)
        return ret / maxreturn, stats
    end
end

function average_normalized_returns_and_action_stats(policy, wrapper, num_trajectories)
    ret = zeros(num_trajectories)
    actions = Int[]
    for idx = 1:num_trajectories
        PPO.reset!(wrapper)
        r, s = single_trajectory_normalized_return_and_action_stats(policy, wrapper)
        ret[idx] = r
        append!(actions, s)
    end
    mean = Flux.mean(ret)
    std = Flux.std(ret)
    stats = [count(actions .== i) for i = 1:NUM_ACTIONS_PER_EDGE]

    return mean, std, stats
end

function single_trajectory_normalized_return(policy, wrapper)
    maxreturn = wrapper.current_score - wrapper.opt_score
    if maxreturn == 0
        return 1.0
    else
        ret = PPO.single_trajectory_return(policy, wrapper)
        return ret / maxreturn
    end
end

function average_normalized_returns(policy, wrapper, num_trajectories)
    ret = zeros(num_trajectories)
    for idx = 1:num_trajectories
        PPO.reset!(wrapper)
        ret[idx] = single_trajectory_normalized_return(policy, wrapper)
    end
    return Flux.mean(ret), Flux.std(ret)
end

function best_single_trajectory_return(policy, wrapper)
    done = PPO.is_terminal(wrapper)

    initial_score = wrapper.current_score
    minscore = wrapper.current_score

    while !done
        state = PPO.state(wrapper) |> gpu
        probs = PPO.action_probabilities(policy, state) |> cpu
        action = rand(Categorical(probs))

        PPO.step!(wrapper, action)

        minscore = min(minscore, wrapper.current_score)
        done = PPO.is_terminal(wrapper)
    end
    return initial_score - minscore
end

function average_best_returns(policy, wrapper, num_trajectories)
    ret = zeros(num_trajectories)
    for idx = 1:num_trajectories
        PPO.reset!(wrapper)
        ret[idx] = best_single_trajectory_return(wrapper, policy)
    end
    return Flux.mean(ret), Flux.std(ret)
end

function best_normalized_single_trajectory_return(policy, wrapper)
    max_return = wrapper.current_score - wrapper.opt_score
    if max_return == 0
        return 1.0
    else
        ret = best_single_trajectory_return(policy, wrapper)
        return ret/max_return
    end
end

function average_normalized_best_returns(policy, wrapper, num_trajectories)
    ret = zeros(num_trajectories)
    for idx = 1:num_trajectories
        PPO.reset!(wrapper)
        ret[idx] = best_normalized_single_trajectory_return(policy, wrapper)
    end
    return Flux.mean(ret), Flux.std(ret)
end

function best_state_in_rollout(wrapper, policy)
    best_wrapper = deepcopy(wrapper)
    minscore = wrapper.current_score
    done = PPO.is_terminal(wrapper)

    while !done
        probs = PPO.action_probabilities(policy, PPO.state(wrapper))
        action = rand(Categorical(probs))

        PPO.step!(wrapper, action)
        done = PPO.is_terminal(wrapper)

        if wrapper.current_score < minscore 
            minscore = wrapper.current_score
            best_wrapper = deepcopy(wrapper)
        end
    end

    return best_wrapper
end
#####################################################################################################################