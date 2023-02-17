using BenchmarkTools
using CUDA


function select(vector, indices)
    return vector[indices]
end

numvals = 10000
vector = rand(numvals)
indices = rand(1:numvals, numvals)

gpu_vector = cu(vector)
gpu_indices = cu(indices)

select(gpu_vector, indices)
select(gpu_vector, indices)
@benchmark CUDA.@sync select($gpu_vector, $indices)

select(gpu_vector, gpu_indices)
select(gpu_vector, gpu_indices)
@benchmark CUDA.@sync select($gpu_vector, $gpu_indices)