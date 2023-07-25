using PlotQuadMesh
using PyPlot
PQ = PlotQuadMesh
#####################################################################################################################
# PLOTTING STUFF
function plot_env_score!(ax, score; coords = (0.8, 0.8), fontsize = 50)
    tpars = Dict(
        :color => "black",
        :horizontalalignment => "center",
        :verticalalignment => "center",
        :fontsize => fontsize,
        :fontweight => "bold",
    )

    ax.text(coords[1], coords[2], score; tpars...)
end

function plot_env(
    env, 
    score, 
    number_elements = false, 
    internal_order = false,
    mark_geometric_vertices=false,
)
    env = deepcopy(env)

    QM.reindex_game_env!(env)
    mesh = env.mesh
    vs = QM.active_vertex_score(env)

    fig, ax = PQ.plot_mesh(
        QM.active_vertex_coordinates(mesh),
        QM.active_quad_connectivity(mesh),
        vertex_score=vs,
        vertex_size = 30,
        number_elements = number_elements,
        internal_order = internal_order,
    )
    
    plot_env_score!(ax, score)

    return fig, ax
end

function plot_wrapper(
    wrapper, 
    filename = ""; 
    xlim=nothing,
    ylim=nothing,
    smooth_iterations = 5, 
    number_elements = false, 
    mark_geometric_vertices = false
    )
    smooth_wrapper!(wrapper, smooth_iterations)

    text = string(wrapper.current_score) * " / " * string(wrapper.opt_score)

    internal_order = number_elements
    element_numbers = number_elements ? findall(wrapper.env.mesh.active_quad) : false

    fig, ax = plot_env(
        wrapper.env, 
        text, 
        element_numbers, 
        internal_order, 
        mark_geometric_vertices
    )

    if isnothing(xlim)
        ax.set_xlim(-1, 1)
    else
        ax.set_xlim(xlim...)
    end

    if isnothing(ylim)
        ax.set_ylim(-1, 1)
    else
        ax.set_ylim(ylim...)
    end

    if length(filename) > 0
        fig.tight_layout()
        fig.savefig(filename)
    end

    return fig
end

function smooth_wrapper!(wrapper, num_iterations = 1)
    for iteration in 1:num_iterations
        QM.averagesmoothing!(wrapper.env.mesh)
    end
end

function plot_trajectory(policy, wrapper, root_directory)
    if !isdir(root_directory)
        mkpath(root_directory)
    end

    fig_name = "figure-" * lpad(0, 3, "0") * ".png"
    filename = joinpath(root_directory, fig_name)
    plot_wrapper(wrapper, filename)

    fig_index = 1
    done = PPO.is_terminal(wrapper)
    while !done 
        probs = PPO.action_probabilities(policy, PPO.state(wrapper))
        action = rand(Categorical(probs))

        PPO.step!(wrapper, action)
        
        fig_name = "figure-" * lpad(fig_index, 3, "0") * ".png"
        filename = joinpath(root_directory, fig_name)
        plot_wrapper(wrapper, filename)
        fig_index += 1

        done = PPO.is_terminal(wrapper)
    end
end

function plot_returns(ret, lower_fill, upper_fill)
    fig, ax = subplots()
    ax.plot(ret)
    ax.fill_between(1:length(ret), lower_fill, upper_fill, alpha = 0.2, facecolor = "blue")
    ax.grid()
    ax.set_xlabel("PPO Iterations")
    ax.set_ylabel("Normalized returns")
    return fig, ax
end

function plot_normalized_returns(ret, dev)
    lower = ret - dev
    upper = ret + dev
    upper[upper .> 1.0] .= 1.0
    return plot_returns(ret, lower, upper)
end

#####################################################################################################################
