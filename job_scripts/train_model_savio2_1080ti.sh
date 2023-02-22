#!/bin/bash
# Job name:
#SBATCH --job-name=Train-Policy-128
#
# Account:
#SBATCH --account=fc_dgflow
#
# Partition:
#SBATCH --partition=savio2_1080ti
#
# QoS:
#SBATCH --qos=savio_normal
#
# Number of nodes:
#SBATCH --nodes=1
#
# Number of tasks (one for each GPU desired for use case) (example):
#SBATCH --ntasks-per-node=1
#
# Processors per task (please always specify the total number of processors twice the number of GPUs):
#SBATCH --cpus-per-task=2
#
#Number of GPUs, this can be in the format of "gpu:[1-4]", or "gpu:K80:[1-4] with the type included
#SBATCH --gres=gpu:1
#
# Wall clock limit:
#SBATCH --time=24:00:00
#
## Command(s) to run (example):

export JULIA_DEPOT_PATH=/global/scratch/users/arjunnarayanan/environments/.julia
module load julia

