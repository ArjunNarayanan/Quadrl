using BSON
include("../src/quad_game_utilities.jl")
using PyPlot


saved_data = "output/model-2/evaluator.bson"
data = BSON.load(saved_data)[:data]
evaluator = data["evaluator"]

mean_returns = evaluator.mean_returns
dev = evaluator.std_returns

lower_bound = mean_returns - dev
upper_bound = mean_returns + dev

fig, ax = subplots()
ax.plot(mean_returns)
ax.fill_between(1:length(mean_returns),lower_bound, upper_bound, alpha = 0.4)
ax.grid()
ax.set_ylim([0.,1.])
ax.set_xlabel("Epochs")
ax.set_ylabel("Mean returns")
fig
fig.savefig("output/model-2/returns.png")