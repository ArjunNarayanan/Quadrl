#####################################################################################################################
# TEMPLATE GENERATION
function pad_matrix_cols(mat, num_new_cols, value)
    nr, _ = size(mat)
    return [mat fill(value, (nr, num_new_cols))]
end

function zero_pad_matrix_cols(m, num_new_cols)
    T = eltype(m)
    return pad_matrix_cols(m, num_new_cols, zero(T))
end

function cycle_edges(x)
    nf, na = size(x)
    x = reshape(x, nf, 4, :)

    x1 = reshape(x, 4nf, 1, :)
    x2 = reshape(x[:, [2, 3, 4, 1], :], 4nf, 1, :)
    x3 = reshape(x[:, [3, 4, 1, 2], :], 4nf, 1, :)
    x4 = reshape(x[:, [4, 1, 2, 3], :], 4nf, 1, :)

    x = cat(x1, x2, x3, x4, dims=2)
    x = reshape(x, 4nf, :)

    return x
end

function make_level4_template(pairs, x)
    cx = cycle_edges(x)

    pcx = zero_pad_matrix_cols(cx, 1)[:, pairs][3:end, :]
    cpcx = cycle_edges(pcx)

    pcpcx = zero_pad_matrix_cols(cpcx, 1)[:, pairs][3:end, :]
    cpcpcx = cycle_edges(pcpcx)

    pcpcpcx = zero_pad_matrix_cols(cpcpcx, 1)[:, pairs][7:end, :]
    cpcpcpcx = cycle_edges(pcpcpcx)

    template = vcat(cx, cpcx, cpcpcx, cpcpcpcx)

    return template
end


function level4_active_template(wrapper)
    @assert !wrapper_requires_reindex(wrapper)
    num_quads = QM.number_of_quads(wrapper.env.mesh)
    active_half_edges = 1:4*num_quads
    template = QM.make_level4_template(wrapper.env.mesh)
    active_template = template[:, active_half_edges]
    return active_template
end

#####################################################################################################################


#####################################################################################################################
# GENERATING AND MANIPULATING ENVIRONMENT STATE
struct StateData
    vertex_score::Any
    action_mask
    optimum_return
    env
end

function Base.show(io::IO, s::StateData)
    println(io, "StateData")
end

function Flux.gpu(s::StateData)
    return StateData(
        gpu(s.vertex_score), 
        gpu(s.action_mask), 
        s.optimum_return,
        s.env
    )
end

function Flux.cpu(s::StateData)
    return StateData(
        cpu(s.vertex_score), 
        cpu(s.action_mask), 
        s.optimum_return,
        s.env
    )
end

function pad_vertex_scores(vertex_scores_vector)
    num_half_edges = [size(vs, 2) for vs in vertex_scores_vector]
    max_num_half_edges = maximum(num_half_edges)
    num_new_cols = max_num_half_edges .- num_half_edges
    padded_vertex_scores = [QM.zero_pad_matrix_cols(vs, nc) for (vs, nc) in zip(vertex_scores_vector, num_new_cols)]
    return padded_vertex_scores
end

function pad_action_mask(action_mask_vector)
    num_actions = length.(action_mask_vector)
    max_num_actions = maximum(num_actions)
    num_new_actions = max_num_actions .- num_actions
    padded_action_mask = [QM.pad_vector(am, nr, -Inf32) for (am, nr) in zip(action_mask_vector, num_new_actions)]
    return padded_action_mask
end

function prepare_state_data_for_batching(state_data_vector)
    vertex_score = [s.vertex_score for s in state_data_vector]
    action_mask = [s.action_mask for s in state_data_vector]

    padded_vertex_scores = pad_vertex_scores(vertex_score)
    padded_action_mask = pad_action_mask(action_mask)
    opt_return = [s.optimum_return for s in state_data_vector]
    envs = [s.env for s in state_data_vector]

    state_data = [StateData(vs, am, opt, env) for (vs, am, opt, env) in zip(padded_vertex_scores, padded_action_mask, opt_return, envs)]
    return state_data
end

function PPO.batch_state(state_data_vector)
    state_data_vector = prepare_state_data_for_batching(state_data_vector)

    vs = [s.vertex_score for s in state_data_vector]
    am = [s.action_mask for s in state_data_vector]
    opt_return = [s.optimum_return for s in state_data_vector]
    envs = [s.env for s in state_data_vector]

    batch_vertex_score = cat(vs..., dims=3)
    batch_action_mask = cat(am..., dims=2)
    return StateData(batch_vertex_score, batch_action_mask, opt_return, envs)
end

function val_or_missing(vector, template, missing_val)
    return [t == 0 ? missing_val : vector[t] for t in template]
end

function action_mask_value(flag)
    if flag
        return -Inf32
    else
        return 0.0f0
    end
end

function action_mask(template)
    requires_mask = mapslices(x -> all(x .== 0), template, dims=1)
    requires_mask = repeat(requires_mask, inner=(NUM_ACTIONS_PER_EDGE, 1))
    requires_mask = vec(requires_mask)
    mask = action_mask_value.(requires_mask)
    return mask
end

function wrapper_requires_reindex(wrapper)
    return (QM.number_of_quads(wrapper.env.mesh) < wrapper.env.mesh.new_quad_pointer - 1) ||
           (QM.number_of_vertices(wrapper.env.mesh) < wrapper.env.mesh.new_vertex_pointer - 1)
end

function PPO.state(wrapper)
    if wrapper_requires_reindex(wrapper)
        QM.reindex_game_env!(wrapper.env)
    end

    env = wrapper.env
    template = level4_active_template(wrapper)

    am = action_mask(template)
    @assert all(am .== 0.0f0)

    vertex_score = Float32.(QM.active_vertex_score(env))
    push!(vertex_score, 0.0f0)
    vertex_degree = Float32.(QM.active_vertex_degrees(env.mesh))
    push!(vertex_degree, 0.0f0)
    @assert length(vertex_score) == length(vertex_degree)

    missing_index = length(vertex_score)
    template[template.==0] .= missing_index
    vs = vertex_score[template]
    vd = vertex_degree[template]

    matrix = vcat(vs, vd)
    opt_return = wrapper.current_score - wrapper.opt_score

    s = StateData(matrix, am, opt_return, wrapper)

    return s
end

function PPO.number_of_actions_per_state(state)
    vs = state.vertex_score
    am = state.action_mask
    @assert ndims(vs) == 3
    @assert ndims(am) == 2
    @assert size(vs, 2) * NUM_ACTIONS_PER_EDGE == size(am, 1)
    return size(am, 1)
end

function PPO.batch_advantage(state, returns)
    return returns ./ state.optimum_return
end
#####################################################################################################################
