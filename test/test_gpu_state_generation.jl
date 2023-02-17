include("../src/quad_game_utilities.jl")
using BenchmarkTools

mesh = QM.square_mesh(100)
edge_pairs = QM.make_edge_pairs(mesh)
edge_ids = reshape(mesh.connectivity, 1, :)

gpu_edge_pairs = edge_pairs |> gpu
gpu_edge_ids = edge_ids |> gpu 


template = make_level4_template(edge_pairs, edge_ids)
template = make_level4_template(edge_pairs, edge_ids)

@benchmark make_level4_template($edge_pairs, $edge_ids)

template = make_level4_template(gpu_edge_pairs, gpu_edge_ids)
template = make_level4_template(gpu_edge_pairs, gpu_edge_ids)

@benchmark CUDA.@sync make_level4_template($gpu_edge_pairs, $gpu_edge_ids)