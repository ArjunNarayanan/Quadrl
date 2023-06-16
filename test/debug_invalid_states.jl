using BSON
include("../src/quad_game_utilities.jl")
include("../random_polygon_environment.jl")
include("../src/plot.jl")



root_dir = "output/model-1/data/states/"
sample_num = 1937

sample_path = joinpath(root_dir, "sample_"*string(sample_num)*".bson")
sample = BSON.load(sample_path)
wrapper = sample[:state].env
mesh = wrapper.env.mesh


@assert length(unique(mesh.q2q[:,4])) > 2

action = 76
quad, edge, type = index_to_action(action)
assert_valid_mesh(wrapper.env.mesh)
PPO.step!(wrapper, action)

# for sample_num=1960:-1:1
#     sample_path = joinpath(root_dir, "sample_"*string(sample_num)*".bson")
#     sample = BSON.load(sample_path)
#     wrapper = sample[:state].env
#     mesh = wrapper.env.mesh
#     if length(unique(mesh.q2q[:,4])) > 2
#         println("FOUND : ", sample_num)
#         break
#     end
# end

# PPO.step!(wrapper, action)
# fig = plot_wrapper(wrapper)