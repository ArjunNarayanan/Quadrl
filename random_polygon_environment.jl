function initialize_random_mesh(poly_degree, quad_alg, round_desired_degree)
    boundary_pts = RQ.random_polygon(poly_degree)
    angles = QM.polygon_interior_angles(boundary_pts)

    bdry_d0 = round_desired_degree ? QM.rounded_desired_degree.(angles) : QM.continuous_desired_degree.(angles)

    mesh = RQ.quad_mesh(boundary_pts, algorithm=quad_alg)
    num_vertices = size(mesh.p, 2)
    is_geometric_vertex = falses(num_vertices)
    is_geometric_vertex[1:poly_degree] .= true

    mesh = QM.QuadMesh(mesh.p, mesh.t, is_geometric_vertex = is_geometric_vertex)

    mask = .![trues(poly_degree); falses(mesh.num_vertices - poly_degree)]
    mask = mask .& mesh.vertex_on_boundary[mesh.active_vertex]

    d0 = [bdry_d0; fill(4, mesh.num_vertices - poly_degree)]
    d0[mask] .= 3

    return mesh, d0
end

function _vanilla_global_score(vertex_score)
    return sum(abs.(vertex_score))
end

function _distance_weighted_global_score(vertex_score, distances)
    score = sum(abs.(vertex_score) .* distances)
    return score
end

function global_score(vertex_score)
    return _vanilla_global_score(vertex_score)
end

function optimal_score(vertex_score)
    return abs(sum(vertex_score))
end

function check_terminated(current_score, opt_score, num_actions, max_actions)
    return (num_actions >= max_actions) || (current_score <= opt_score)
end

mutable struct RandPolyEnv
    poly_degree_list
    poly_degree
    quad_alg
    num_actions
    max_actions_factor
    max_actions::Any
    env::Any
    current_score
    opt_score
    is_terminated
    reward
    cleanup
    round_desired_degree
    function RandPolyEnv(
        poly_degree_list, 
        max_actions_factor, 
        quad_alg, 
        cleanup,
        round_desired_degree
    )
        @assert max_actions_factor > 0
        @assert all(poly_degree_list .> 3)

        poly_degree = rand(poly_degree_list)
        max_actions = max_actions_factor*poly_degree
        mesh, d0 = initialize_random_mesh(poly_degree, quad_alg, round_desired_degree)
        env = QM.GameEnv(mesh, d0)
        current_score = global_score(env.vertex_score)
        opt_score = optimal_score(env.vertex_score)
        reward = 0.0f0
        num_actions = 0
        is_terminated = check_terminated(current_score, opt_score, num_actions, max_actions)
        new(
            poly_degree_list,
            poly_degree, 
            quad_alg, 
            num_actions,
            max_actions_factor, 
            max_actions, 
            env, 
            current_score, 
            opt_score, 
            is_terminated, 
            reward, 
            cleanup,
            round_desired_degree
        )
    end
end

function Base.show(io::IO, wrapper::RandPolyEnv)
    println(io, "RandPolyEnv")
    println(io, "\t$(wrapper.poly_degree) polygon degree")
    println(io, "\t$(wrapper.max_actions) max actions")
    show(io, wrapper.env)
end

function PPO.reset!(wrapper::RandPolyEnv)
    wrapper.poly_degree = rand(wrapper.poly_degree_list)
    wrapper.max_actions = wrapper.max_actions_factor * wrapper.poly_degree
    mesh, d0 = initialize_random_mesh(
        wrapper.poly_degree, 
        wrapper.quad_alg,
        wrapper.round_desired_degree
    )
    wrapper.env = QM.GameEnv(mesh, d0)
    wrapper.current_score = global_score(wrapper.env.vertex_score)
    wrapper.reward = 0
    wrapper.num_actions = 0
    wrapper.opt_score = optimal_score(wrapper.env.vertex_score)
    wrapper.is_terminated = check_terminated(wrapper.current_score, wrapper.opt_score,
        wrapper.num_actions, wrapper.max_actions)
end

function update_env_after_step!(wrapper)
    if wrapper.cleanup
        maxsteps = 2 * QM.number_of_quads(wrapper.env.mesh)
        QM.cleanup_env!(wrapper.env, maxsteps)
    end

    wrapper.current_score = global_score(wrapper.env.vertex_score)
    wrapper.num_actions += 1
    wrapper.is_terminated = check_terminated(
        wrapper.current_score, 
        wrapper.opt_score, 
        wrapper.num_actions, 
        wrapper.max_actions
    )
end
