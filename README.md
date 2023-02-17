# Quadrl

Optimizing quadrilateral meshes with reinforcement learning.

## Setting up environment on Savio

Ensure that `JULIA_DEPOT_PATH` and `TMP` point to somewhere on your scratch folder otherwise you will get a system out-of-error message.

```
export JULIA_DEPOT_PATH=/global/scratch/users/<user_name>/path/to/.julia
export TMP=/global/scratch/users/<user_name>/path/to/tmp
module load julia
```

```julia
using Pkg
Pkg.add("Flux")
```