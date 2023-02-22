include("src/quad_game_utilities.jl")
include("random_polygon_environment.jl")


discount = 1.0
epsilon = 0.05f0
minibatch_size = 64
episodes_per_iteration = 200
epochs_per_iteration = 20
num_iter = 2000
quad_alg = "catmull-clark"
root_dir = "/global/home/users/arjunnarayanan/Research/MeshRL/Quadrl/"

num_evaluation_trajectories = 100
output_dir = joinpath(root_dir, "output")
evaluator = SaveBestModel(output_dir, num_evaluation_trajectories)

poly_degree = 10
max_actions = 20
wrapper = RandPolyEnv(poly_degree, max_actions, quad_alg)

policy = SimplePolicy.Policy(216, 128, 5, 5) |> gpu
optimizer = ADAM(1e-4)

PPO.ppo_iterate!(policy, 
                 wrapper, 
                 optimizer,
                 episodes_per_iteration, 
                 minibatch_size, 
                 num_iter, 
                 evaluator,
                 epochs_per_iteration,
                 discount,
                 epsilon)