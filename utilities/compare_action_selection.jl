using BSON
using TOML
include("../src/quad_game_utilities.jl")
using PyPlot



function load_evaluator(model_name)
    input_dir = joinpath("output", model_name)
    data_filename = joinpath(input_dir, "evaluator.bson")
    data = BSON.load(data_filename)[:data]
    return data
end

function plot_evaluator(evaluator, title="")
    fig, ax = subplots()
    action_counts = transpose(
        hcat(
            evaluator.action_counts...
        )
    )
    for action_type in 1:5    
        ax.plot(
            action_counts[:,action_type], 
            label="type "*string(action_type)
            )
    end
    ax.legend()
    ax.set_title(title)
    ax.grid()
    return fig
end


discrete_model_name = "model-2"
discrete_evaluator= load_evaluator(model_discrete_name)["evaluator"];

continuous_model_name = "model-3"
continuous_evaluator= load_evaluator(continuous_model_name)["evaluator"];

fig = plot_evaluator(
    discrete_evaluator,
    "discrete"
)
fig.savefig("output/comparison/discrete_actions.png")

fig = plot_evaluator(
    continuous_evaluator,
    "continuous"
)
fig.savefig("output/comparison/continuous_actions.png")