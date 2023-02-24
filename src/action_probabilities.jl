#####################################################################################################################
# EVALUATING POLICY
function PPO.action_probabilities(policy, state)
    # @assert policy.num_output_channels == NUM_ACTIONS_PER_EDGE
    
    vertex_score, action_mask = state.vertex_score, state.action_mask
    logits = vec(policy(vertex_score)) + action_mask
    p = softmax(logits)
    p = p

    return p
end

function PPO.batch_action_probabilities(policy, state)
    # @assert policy.num_output_channels == NUM_ACTIONS_PER_EDGE

    vertex_score, action_mask = state.vertex_score, state.action_mask
    nf, nq, nb = size(vertex_score)
    logits = reshape(policy(vertex_score), :, nb) + action_mask
    probs = softmax(logits, dims=1)
    return probs
end
#####################################################################################################################
