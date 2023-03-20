using BSON
include("../src/quad_game_utilities.jl")
using PyPlot


saved_data = "output/model-1/loss.bson"
data = BSON.load(saved_data)[:loss]

ppo_loss = data["ppo"]
entropy_loss = data["entropy"]

fig, ax = subplots()
ax.plot(ppo_loss, label="ppo")
# ax.plot(entropy_loss, label="entropy")
ax.legend()
ax.grid()
fig

fig, ax = subplots()
ax.plot(entropy_loss)
ax.grid()
fig