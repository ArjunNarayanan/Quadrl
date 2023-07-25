#####################################################################################################################
# STEPPING THE ENVIRONMENT
function PPO.reward(wrapper)
    return wrapper.reward
end

function PPO.is_terminal(wrapper)
    return wrapper.is_terminated
end

function index_to_action(index)
    actions_per_quad = NUM_EDGES_PER_ELEMENT * NUM_ACTIONS_PER_EDGE

    quad = div(index - 1, actions_per_quad) + 1

    quad_action_idx = rem(index - 1, actions_per_quad)
    edge = div(quad_action_idx, NUM_ACTIONS_PER_EDGE) + 1
    action = rem(quad_action_idx, NUM_ACTIONS_PER_EDGE) + 1

    return quad, edge, action
end

function action_space_size(env)
    nq = QM.quad_buffer(env.mesh)
    return nq * 4 * NUM_ACTIONS_PER_EDGE
end

function assert_valid_mesh(mesh)
    @assert QM.all_active_vertices(mesh) "Found inactive vertices in mesh connectivity"
    @assert QM.no_quad_self_reference(mesh) "Found self-referencing quads in mesh q2q"
    @assert QM.all_active_quad_or_boundary(mesh) "Found inactive quads in mesh q2q"
end

function is_valid_mesh(mesh)
    return QM.all_active_vertices(mesh) && 
    QM.no_quad_self_reference(mesh) &&
    QM.all_active_quad_or_boundary(mesh)
end

function terminate_invalid_environment(wrapper)
    opt_return = wrapper.current_score - wrapper.opt_score
    # set the reward such that the normalized reward is -1
    wrapper.reward = -1.0 * opt_return
    wrapper.is_terminated = true
end

function step_wrapper!(wrapper, quad, edge, type)
    env = wrapper.env
    previous_score = wrapper.current_score
    success = false

    # @assert QM.is_active_quad(env.mesh, quad) "Attempting to act on inactive quad $quad with action ($quad, $edge, $type)"
    @assert type in 1:NUM_ACTIONS_PER_EDGE "Expected action type in {1,2,3,4} got type = $type"
    @assert edge in (1, 2, 3, 4) "Expected edge in {1,2,3,4} got edge = $edge"
    # @assert wrapper.opt_score == optimal_score(wrapper.env.vertex_score)


    if !is_valid_mesh(env.mesh)
        terminate_invalid_environment(wrapper)
        return
    elseif !QM.is_active_quad(env.mesh, quad)
        success = false
    elseif type == 1
        success = QM.step_left_flip!(env, quad, edge)
    elseif type == 2
        success = QM.step_right_flip!(env, quad, edge)
    elseif type == 3
        success = QM.step_split!(env, quad, edge)
    elseif type == 4
        success = QM.step_collapse!(env, quad, edge)
    elseif type == 5
        maxsplits = 2*QM.number_of_quads(env.mesh)
        success = QM.step_global_split_without_loops!(env, quad, edge, maxsplits)
    else
        error("Unexpected action type $type")
    end

    update_env_after_step!(wrapper)
    
    if success
        wrapper.reward = previous_score - wrapper.current_score
    else
        wrapper.reward = NO_ACTION_REWARD
    end
    
end

function PPO.step!(wrapper, action_index)
    env = wrapper.env
    na = action_space_size(env)
    @assert 0 < action_index <= na "Expected 0 < action_index <= $na, got action_index = $action_index"
    @assert !wrapper.is_terminated "Attempting to step in terminated environment with action $action_index"

    quad, edge, type = index_to_action(action_index)
    step_wrapper!(wrapper, quad, edge, type)
end
#####################################################################################################################
