from collections import defaultdict
import torch
import time

#from scripts.plan import start_time

REWARD_DIM = VALUE_DIM = 1
#from .mcts import *
#from .mcts_beam import *
from .mcts_expand import *
import networkx as nx
import matplotlib.pyplot as plt
#from collections import Counter
import torch.nn.functional as F
import time

@torch.no_grad()
def model_rollout_continuous(model, x, latent, denormalize_rew, denormalize_val, discount, prob_penalty_weight=1e4):
    prediction = model.decode(latent, x[:, -1, :model.observation_dim])
    prediction = prediction.reshape([-1, model.transition_dim])

    r_t, V_t = prediction[:, -3], prediction[:, -2]
    if denormalize_rew is not None:
        r_t = denormalize_rew(r_t).reshape([x.shape[0], -1])
    if denormalize_val is not None:
        V_t = denormalize_val(V_t).reshape([x.shape[0], -1])

    # discounts with terminal flag
    terminal = prediction[:, -1].reshape([x.shape[0], -1])
    discounts = torch.cumprod(torch.ones_like(r_t) * discount * (1-terminal), dim=-1)
    values = torch.sum(r_t[:,:-1] * discounts[:, :-1], dim=-1) + V_t[:,-1] * discounts[:,-1]
    prob_penalty = prob_penalty_weight * torch.mean(torch.square(latent), dim=-1)
    objective = values - prob_penalty
    return objective.cpu().numpy(), prediction.cpu().numpy()


import numpy as np


@torch.no_grad()
def sample(model, x, denormalize_rew, denormalize_val, discount, steps, nb_samples=4096, rounds=8):
    indicies = torch.randint(0, model.model.K-1, size=[nb_samples, steps // model.latent_step],
                             device=x.device, dtype=torch.int32)
    prediction_raw = model.decode_from_indices(indicies, x[:, 0, :model.observation_dim])
    prediction = prediction_raw.reshape([-1, model.transition_dim])

    r_t, V_t = prediction[:, -3], prediction[:, -2]
    if denormalize_rew is not None:
        r_t = denormalize_rew(r_t).reshape([indicies.shape[0], -1])
    if denormalize_val is not None:
        V_t = denormalize_val(V_t).reshape([indicies.shape[0], -1])

    discounts = torch.cumprod(torch.ones_like(r_t) * discount, dim=-1)
    values = torch.sum(r_t[:,:-1] * discounts[:, :-1], dim=-1) + V_t[:,-1]*discounts[:, -1]
    optimal = prediction_raw[values.argmax()]
    print(values.max().item())
    return optimal.cpu().numpy()


@torch.no_grad()
def sample_with_prior(prior, model, x, denormalize_rew, denormalize_val, discount, steps, nb_samples=4096, rounds=8,
                      likelihood_weight=5e2, prob_threshold=0.05, uniform=False, return_info=False):
    state = x[:, 0, :model.observation_dim]
    optimals = []
    optimal_values = []
    info = defaultdict(list)
    for round in range(rounds):
        contex = None
        acc_probs = torch.zeros([1]).to(x)
        for step in range(steps//model.latent_step):
            logits, _ = prior(contex, state) # [B x t x K]
            probs = raw_probs = torch.softmax(logits[:, -1, :], dim=-1) # [B x K]
            log_probs = torch.log(probs)
            if uniform:
                valid = probs > 0
                probs = valid/valid.sum(dim=-1)[:, None]
            if step == 0:
                samples = torch.multinomial(probs, num_samples=nb_samples//rounds, replacement=True) # [B, M]
            else:
                samples = torch.multinomial(probs, num_samples=1, replacement=True)  # [B, M]
            samples_prob = torch.cat([torch.index_select(a, 0, i).unsqueeze(0) for a, i in zip(log_probs, samples)]) # [B, M]
            acc_probs = acc_probs + samples_prob.reshape([-1])
            if not contex is None:
                contex = torch.cat([contex, samples.reshape([-1, 1])], dim=1)
            else:
                contex = samples.reshape([-1, step+1]) # [(B*M) x t]
        prediction_raw = model.decode_from_indices(contex, state)
        prediction = prediction_raw.reshape([-1, model.transition_dim])

        r_t = prediction[:, -3]
        V_t = prediction[:, -2]
        terminals = prediction[:, -1].reshape([contex.shape[0], -1])
        if denormalize_rew is not None:
            r_t = denormalize_rew(r_t).reshape([contex.shape[0], -1])
        if denormalize_val is not None:
            V_t = denormalize_val(V_t).reshape([contex.shape[0], -1])

        discounts = torch.cumprod(torch.ones_like(r_t) * discount, dim=-1)
        values = torch.sum(r_t[:,:-1] * discounts[:, :-1], dim=-1) + V_t[:,-1] * discounts[:,-1]
        likelihood_bonus = likelihood_weight*torch.clamp(acc_probs, -1e5, np.log(prob_threshold)*(steps//model.latent_step))
        info["log_probs"].append(acc_probs.cpu().numpy())
        info["returns"].append(values.cpu().numpy())
        info["predictions"].append(prediction_raw.cpu().numpy())
        info["objectives"].append(values.cpu().numpy() + likelihood_bonus.cpu().numpy())
        info["latent_codes"].append(contex.cpu().numpy())
        max_idx = (values+likelihood_bonus).argmax()
        optimal_value = values[max_idx]
        optimal = prediction_raw[max_idx]
        optimals.append(optimal)
        optimal_values.append(optimal_value.item())

    for key, val in info.items():
        info[key] = np.concatenate(val, axis=0)
    max_idx = np.array(optimal_values).argmax()
    optimal = optimals[max_idx]
    print(f"predicted max value {optimal_values[max_idx]}")
    if return_info:
        return optimal.cpu().numpy(), info
    else:
        return optimal.cpu().numpy()


@torch.no_grad()
def sample_with_prior_tree(prior, model, x, denormalize_rew, denormalize_val, discount, steps, samples_per_latent=16, likelihood_weight=0.0):
    contex = None
    state = x[:, 0, :model.observation_dim]
    acc_probs = torch.ones([1]).to(x)
    for step in range(steps//model.latent_step):
        logits, _ = prior(contex, state) # [B x t x K]
        probs = torch.softmax(logits[:, -1, :], dim=-1) # [B x K]
        samples = torch.multinomial(probs, num_samples=samples_per_latent, replacement=True) # [B, M]
        samples_prob = torch.cat([torch.index_select(a, 0, i).unsqueeze(0) for a, i in zip(probs, samples)]) # [B, M]
        acc_probs = acc_probs.repeat_interleave(samples_per_latent, 0) * samples_prob.reshape([-1])
        if not contex is None:
            contex = torch.cat([torch.repeat_interleave(contex, samples_per_latent, 0), samples.reshape([-1, 1])],
                               dim=1)
        else:
            contex = samples.reshape([-1, step+1]) # [(B*M) x t]

    prediction_raw = model.decode_from_indices(contex, state)
    prediction = prediction_raw.reshape([-1, model.transition_dim])

    r_t, V_t = prediction[:, -3], prediction[:, -2]


    if denormalize_rew is not None:
        r_t = denormalize_rew(r_t).reshape([contex.shape[0], -1])
    if denormalize_val is not None:
        V_t = denormalize_val(V_t).reshape([contex.shape[0], -1])

    discounts = torch.cumprod(torch.ones_like(r_t) * discount, dim=-1)
    values = torch.sum(r_t[:,:-1] * discounts[:, :-1], dim=-1) + V_t[:,-1]*discounts[:, -1]
    likelihood_bouns = likelihood_weight*torch.log(acc_probs)
    max_idx = (values+likelihood_bouns).argmax()
    optimal = prediction_raw[max_idx]
    print(f"predicted max value {values[max_idx]}, likelihood {acc_probs[max_idx]} with bouns {likelihood_bouns[max_idx]}")
    return optimal.cpu().numpy()

@torch.no_grad()
def beam_with_prior(prior, model, x, denormalize_rew, denormalize_val, discount, steps,
                    beam_width, n_expand, n_action, b_percent, action_percent,
                    pw_alpha, mcts_itr, prob_threshold=0.05, likelihood_weight=5e2, prob_acc="product", return_info=False):
    state = x[:, 0, :prior.observation_dim]
    def tensor_to_tuple(tensor):
        return tuple(tensor.cpu().numpy().flatten())

    # Initialize the outer dictionary
    state_dict = {}
    # Store the value in the nested dictionary
    def store_value(state, action_matrix, index):
        state_key = tensor_to_tuple(state)
        if state_key not in state_dict:
            state_dict[state_key] = [action_matrix,index]
    #for step in range(steps//model.latent_step):
    import time
    start = time.time()
    max_depth = 3
    tree_gamma = 0.99
    action_sequence = 3
    mse_factor = 0
    for step in range(max_depth):
        if step == 0:
            logits, _ = prior(None, state) # [B x t x K]
        else:
            logits, _ = prior(None, state_for_next_prior) #used to sample intermediate action contex
        action_probs = torch.softmax(logits[:, -1, :], dim=-1) # [B x K]
        nb_samples = beam_width if step == 0 else n_action

        action_samples = torch.multinomial(action_probs, num_samples=nb_samples, replacement=False) # [B, M]
        # Gather the corresponding probabilities for the sampled actions
        action_probs_sampled = torch.gather(action_probs, 1, action_samples)
        action_contex = action_samples.reshape([-1, 1]) # [(B*M) x t]

        if step == 0:
            logits, _ = prior(action_contex, state)
        else:
            history_contex = history_contex.repeat_interleave(nb_samples, 0)
            history_partial = history_contex[:, :-1]  # This will have shape [104, 1]
            # Concatenate along dimension 1 (the columns)
            action_contex = torch.cat([history_partial, action_contex], dim=1)
            repeated_state = state.repeat(action_contex.shape[0], 1)
            #use previous info as contex here
            #context history
            logits, _ = prior(action_contex, repeated_state)
        probs = torch.softmax(logits[:, -1, :], dim=-1)  # [B x K]
        samples = torch.multinomial(probs, num_samples=n_expand, replacement=True)  # [B, M]
        contex = torch.cat([torch.repeat_interleave(action_contex, n_expand, 0), samples.reshape([-1, 1])], dim=1)
        if step == 0:
            prediction_raw = model.decode_from_indices(contex, state)

            #Please ignore this chunk of code
            #following chunk of code is for testing hypothesis of pattern match for autoregressive prediction and short context prediction
            ##################################################################################
            #predicted_second_state_test = prediction_raw[0, 1, 1:1 + model.observation_dim]
            #print("initial:",prediction_raw[0,1, 1 + model.observation_dim:-1])
            #initial = prediction_raw[0,1, 1 + model.observation_dim: 1 + 2*model.observation_dim]
            #print(contex[0][1].unsqueeze(0))
            #print(contex[0].view(1,1), contex[0].view(1,1).shape)
            #print(contex[0][1].view(1,1))
            #print(torch.tensor([[contex[0][1], contex[0][1]]], device='cuda:0').shape)
            #print(predicted_second_state_test.shape)
            # logits, _ = prior(None, predicted_second_state_test.view([1, -1]))  # used to sample intermediate action contex
            # action_probs2 = torch.softmax(logits[:, -1, :], dim=-1)  # [B x K]
            # action_samples2 = torch.multinomial(action_probs2, num_samples=10, replacement=True)  # [B, M]
            # # action contex needs to be concadenated here
            # action_contex2 = action_samples2.reshape([-1, 1])  # [(B*M) x t]
            # # print(action_contex2.shape, contex[0,0])
            # repeated_context = contex[0, 0].unsqueeze(0).unsqueeze(1).expand(action_contex2.shape[0], 1)
            # result = torch.cat([repeated_context, action_contex2], dim=1)
            # initial = model.decode_from_indices(result, state)[:,1,1:1 + model.observation_dim]
            # second = model.decode_from_indices(action_contex2, predicted_second_state_test.view([1, -1]))[:,0,1:1 + model.observation_dim]
            # mse_per_row = ((initial - second)**2).mean(dim=1)
            #print(mse_per_row)
            #print(result)
            # test_second = model.decode_from_indices(torch.tensor([[contex[0][1], contex[0][1]]], device='cuda:0'), predicted_second_state_test.view([1, -1]))
            # test_three = model.decode_from_indices(torch.tensor([[100, 100]], device='cuda:0'),
            #                                         predicted_second_state_test.view([1, -1]))
            #test_four = model.decode_from_indices(action_contex2, predicted_second_state_test.view([1, -1]))
            #initial_expanded = initial.unsqueeze(0).expand(test_four.shape[0], -1)
            #mse_per_row = ((test_four[:, 0, 1:1 + model.observation_dim] - initial_expanded) ** 2).mean(dim=1)
            #prediction_raw = model.decode_from_indices(contex[0][1], state)
            #######################################################################################

            reshaped_prediction_raw = prediction_raw.view(nb_samples, n_expand, 2, -1)
            expanded_action_contex = action_contex.unsqueeze(1).unsqueeze(2).expand(nb_samples, n_expand, 2, 1)
            predicted_first_state = prediction_raw[:, 0, 1:1+model.observation_dim]
            decoded_state_compare = state.expand_as(predicted_first_state)
            mse_loss_per_element = F.mse_loss(predicted_first_state, decoded_state_compare, reduction='none')
            mse_loss_per_example = mse_loss_per_element.mean(dim=1)
            mse_loss_per_example = mse_loss_per_example.view(nb_samples, n_expand)
            expanded_mse_loss = mse_loss_per_example.unsqueeze(2).unsqueeze(3).expand(nb_samples, n_expand, 2, 1)
            expanded_prior_probs = action_probs_sampled.reshape([-1, 1]).unsqueeze(2).unsqueeze(3).expand(nb_samples, n_expand, 2, 1)
            concatenated_tensor = torch.cat([reshaped_prediction_raw, expanded_prior_probs], dim=3)
            concatenated_tensor = torch.cat([concatenated_tensor,expanded_action_contex], dim=3)
            final_tensor = torch.cat([concatenated_tensor, expanded_mse_loss], dim=3)
            expansion_values = final_tensor[:, :, 1, 0]
            action_values = final_tensor[:, 0, 0, 0].view(-1, 1)
            action_mse = final_tensor[:, 0, 0, -1]
            expansion_values *= (tree_gamma ** action_sequence)
            mean_values = torch.cat((expansion_values, action_values), dim=1)
            mean_values = mean_values.mean(dim=1)
            mean_values = mean_values - mse_factor*action_mse

            k = int(mean_values.size(0)*b_percent) if int(mean_values.size(0)*b_percent) >=1 else 1
            values_with_b, index = torch.topk(mean_values, k)
            store_value(state, final_tensor, index)
            state_for_next_prior = final_tensor[index,:,1,1:1+model.observation_dim]
            history_contex = contex.view(nb_samples, n_expand, 2, -1)[index].squeeze(-1)
            history_contex = history_contex.view(-1, history_contex.size(-1))
            original_ctx_dtype = history_contex.dtype
            original_state_dtype = state_for_next_prior.dtype
            #unique_tensor = torch.unique(state_for_next_prior, dim=1)
            state_for_next_prior = state_for_next_prior.view(-1, state_for_next_prior.size(-1))
            combined = torch.cat([history_contex, state_for_next_prior], dim=1)
            unique_combined = torch.unique(combined, dim=0)
            # Split the unique tensor back into the original components
            history_contex = unique_combined[:, :history_contex.size(1)].to(original_ctx_dtype)
            state_for_next_prior = unique_combined[:, history_contex.size(1):].to(original_state_dtype)

        else:
            repeated_state = repeated_state.repeat_interleave(n_expand, 0)
            prediction_raw = model.decode_for_ood(contex, repeated_state)
            prediction_raw = prediction_raw[:,-2:,:]
            reshaped_prediction_raw = prediction_raw.view(-1,nb_samples, n_expand, 2, model.observation_dim + 3*model.action_dim+ 2)
            action_contex = action_contex[:,-1].view(-1, nb_samples, 1)
            action_probs_sampled = action_probs_sampled.view(-1, nb_samples, 1)
            expanded_prior_probs = action_probs_sampled.unsqueeze(3).unsqueeze(4).expand(-1, nb_samples,
                                                                                             n_expand, 2, 1)

            expanded_action_contex = action_contex.unsqueeze(3).unsqueeze(4).expand(-1, nb_samples,
                                                                                             n_expand, 2, 1)

            concatenated_tensor = torch.cat([reshaped_prediction_raw, expanded_prior_probs], dim=4)
            concatenated_tensor = torch.cat([concatenated_tensor,expanded_action_contex], dim=4)
            zero_tensor = torch.zeros(concatenated_tensor.shape[0],
                                      concatenated_tensor.shape[1],
                                      concatenated_tensor.shape[2],
                                      concatenated_tensor.shape[3],
                                      1,
                                      device=concatenated_tensor.device)
            final_tensor = torch.cat([concatenated_tensor, zero_tensor], dim=4)
            expansion_values = final_tensor[:, :, :, 1, 0]
            action_values = final_tensor[:, :, 0, 0, 0].view(state_for_next_prior.shape[0], -1, 1)
            action_mse = final_tensor[:, :, 0, 0, -1]
            expansion_values *= (tree_gamma ** action_sequence)

            #average means
            mean_values = torch.cat((expansion_values, action_values), dim=2)
            mean_values = mean_values.mean(dim=2) -  mse_factor*action_mse

            k = max(int(mean_values.size(1) * action_percent), 1)
            all_selected_tensors = []
            all_selected_history = []
            values_with_b, index = torch.topk(mean_values, k)
            history_contex = contex.view(state_for_next_prior.shape[0], nb_samples, n_expand, -1)
            for i in range(state_for_next_prior.shape[0]):
                store_value(state_for_next_prior[i], final_tensor[i], index[i])
                all_selected_tensors.append(final_tensor[i][index[i]])
                selected_history = history_contex[i][index[i]]
                all_selected_history.append(selected_history)
            final_selected_state = torch.cat(all_selected_tensors, dim=0)
            final_selected_state = final_selected_state.view(-1,final_selected_state.size(2), final_selected_state.size(3))
            final_selected_state = final_selected_state[:,1,1:1+model.observation_dim]
            final_history = torch.cat(all_selected_history, dim=0)
            final_history = final_history.view(final_selected_state.size(0), -1)
            original_ctx_dtype = final_history.dtype
            original_state_dtype = final_selected_state.dtype
            combined = torch.cat([final_history, final_selected_state], dim=1)
            unique_combined = torch.unique(combined, dim=0)
            # # Split the unique tensor back into the original components
            history_contex = unique_combined[:, :final_history.size(1)].to(original_ctx_dtype)
            state_for_next_prior = unique_combined[:, final_history.size(1):].to(original_state_dtype)
    print("inference time,",time.time() - start)
    mcts_instance = MCTS(state, state_dict, tree_gamma, prior, model, int(n_action*action_percent), n_expand, mse_factor, max_depth-1)

    start_time = time.time()
    mcts_instance.search(mcts_itr)
    values_list = list(mcts_instance.Qsa.values())

    # Stack the tensors into one tensor
    values_tensor = torch.stack(values_list)

    # Compute the mean and standard deviation
    value_mean = torch.mean(values_tensor)
    value_std = torch.std(values_tensor)
    value_max = torch.max(values_tensor)
    value_min = torch.min(values_tensor)
    # Print the results
    print("Mean:", value_mean.item(), "Std:", value_std.item(), "Max:", value_max.item(), "Min:", value_min.item())
    # Stop the timer
    end_time = time.time()

    # Calculate the running time
    running_time = end_time - start_time
    print("search time,", running_time)
    best_action = mcts_instance.best_action().long()
    print(best_action)
    prediction_raw = model.decode_from_indices(best_action.view(1, -1), state).squeeze(0)
    return prediction_raw.cpu().numpy()



@torch.no_grad()
def MCTS_P(prior, model, x, denormalize_rew, denormalize_val, discount, steps,
                    beam_width, n_expand, n_action, b_percent, action_percent,
                    pw_alpha, mcts_itr, prob_threshold=0.05, likelihood_weight=5e2, prob_acc="product", return_info=False):
    state = x[:, 0, :prior.observation_dim]
    def tensor_to_tuple(tensor):
        return tuple(tensor.cpu().numpy().flatten())

    # Initialize the cache/ Pre-constructed search space
    state_dict = {}
    # Store the value in the nested dictionary
    def store_value(state, action_matrix, index):
        state_key = tensor_to_tuple(state)
        #print(state_key)
        if state_key not in state_dict:
            state_dict[state_key] = [action_matrix,index]
    import time
    start = time.time()
    max_depth = 3
    tree_gamma = 0.99
    action_sequence = 3
    mse_factor = 0
    for step in range(max_depth):
        if step == 0:
            logits, _ = prior(None, state) # [B x t x K]
        else:
            #contex = None
            logits, _ = prior(None, state_for_next_prior)
        action_probs = torch.softmax(logits[:, -1, :], dim=-1) # [B x K]
        nb_samples = beam_width if step == 0 else n_action

        action_samples = torch.multinomial(action_probs, num_samples=nb_samples, replacement=False) # [B, M]
        # Gather the corresponding probabilities for the sampled actions
        action_probs_sampled = torch.gather(action_probs, 1, action_samples)
        action_contex = action_samples.reshape([-1, 1]) # [(B*M) x t]
        if step == 0:
            logits, _ = prior(action_contex, state)
        else:
            #print(state_for_next_prior.shape)
            state_for_next_prior_expanded = state_for_next_prior.repeat_interleave(nb_samples, 0)
            logits, _ = prior(action_contex, state_for_next_prior_expanded)
        probs = torch.softmax(logits[:, -1, :], dim=-1)  # [B x K]
        log_probs = torch.log(probs)
        samples = torch.multinomial(probs, num_samples=n_expand, replacement=True)  # [B, M]
        contex = torch.cat([torch.repeat_interleave(action_contex, n_expand, 0), samples.reshape([-1, 1])], dim=1)
        print(contex.shape)
        if step == 0:
            prediction_raw = model.decode_from_indices(contex, state)
            #print(prediction_raw.shape)
            reshaped_prediction_raw = prediction_raw.view(nb_samples, n_expand, 2, -1)
            #print("prediction_raw,",prediction_raw.shape)
            #print(action_contex)
            expanded_action_contex = action_contex.unsqueeze(1).unsqueeze(2).expand(nb_samples, n_expand, 2, 1)
            #print("expanded_action_contex,",expanded_action_contex.shape)
            #print(expanded_action_contex)
            predicted_first_state = prediction_raw[:, 0, 1:1+model.observation_dim]
            decoded_state_compare = state.expand_as(predicted_first_state)

            mse_loss_per_element = F.mse_loss(predicted_first_state, decoded_state_compare, reduction='none')
            mse_loss_per_example = mse_loss_per_element.mean(dim=1)
            mse_loss_per_example = mse_loss_per_example.view(nb_samples, n_expand)
            expanded_mse_loss = mse_loss_per_example.unsqueeze(2).unsqueeze(3).expand(nb_samples, n_expand, 2, 1)
            expanded_prior_probs = action_probs_sampled.reshape([-1, 1]).unsqueeze(2).unsqueeze(3).expand(nb_samples, n_expand, 2, 1)
            concatenated_tensor = torch.cat([reshaped_prediction_raw, expanded_prior_probs], dim=3)
            concatenated_tensor = torch.cat([concatenated_tensor,expanded_action_contex], dim=3)
            final_tensor = torch.cat([concatenated_tensor, expanded_mse_loss], dim=3)
            expansion_values = final_tensor[:, :, 1, 0]   #return to go for sampled state
            action_values = final_tensor[:, 0, 0, 0].view(-1, 1)  #return to go for bootstrapping Q value
            action_mse = final_tensor[:, 0, 0, -1]
            print(final_tensor.shape)
            expansion_values *= (tree_gamma ** action_sequence) #short back propagation

            #weighted means
            #expansion_mean = expansion_values.mean(dim=1)
            #action_mean = action_values.mean(dim=1)
            #mean_values = action_mean + 0.1 * (expansion_mean - action_mean)

            mean_values = torch.cat((expansion_values, action_values), dim=1)
            mean_values = mean_values.mean(dim=1)
            mean_values = mean_values - mse_factor*action_mse

            k = int(mean_values.size(0)*b_percent) if int(mean_values.size(0)*b_percent) >=1 else 1
            values_with_b, index = torch.topk(mean_values, k)
            store_value(state, final_tensor, index)

            state_for_next_prior = final_tensor[index,:,1,1:1+model.observation_dim]
            state_for_next_prior = state_for_next_prior.view(-1, state_for_next_prior.size(-1))
            state_for_next_prior = torch.unique(state_for_next_prior, dim=0)

        else:

            state_for_next_prior_expanded = state_for_next_prior_expanded.repeat_interleave(n_expand, 0)
            prediction_raw = model.decode_for_ood(contex, state_for_next_prior_expanded)
            predicted_first_state = prediction_raw[:, 0,1:model.observation_dim+1]
            decoded_state_compare = state_for_next_prior_expanded


            mse_loss_per_element = F.mse_loss(predicted_first_state, decoded_state_compare, reduction='none')
            mse_loss_per_example = mse_loss_per_element.mean(dim=1)
            mse_loss_per_example = mse_loss_per_example.view(-1, nb_samples, n_expand)

            expanded_mse_loss = mse_loss_per_example.unsqueeze(3).unsqueeze(4).expand(-1, nb_samples, n_expand, 2, 1)
            #print(expanded_mse_loss[0])

            reshaped_prediction_raw = prediction_raw.view(-1,nb_samples, n_expand, 2, model.observation_dim + 3*model.action_dim+ 2)
            #print(reshaped_prediction_raw.shape)
            action_contex = action_contex.view(-1, nb_samples, 1)
            action_probs_sampled = action_probs_sampled.view(-1, nb_samples, 1)
            expanded_prior_probs = action_probs_sampled.unsqueeze(3).unsqueeze(4).expand(-1, nb_samples,
                                                                                             n_expand, 2, 1)

            #print("reshaped_prediction_raw", reshaped_prediction_raw.shape)
            expanded_action_contex = action_contex.unsqueeze(3).unsqueeze(4).expand(-1, nb_samples,
                                                                                             n_expand, 2, 1)

            concatenated_tensor = torch.cat([reshaped_prediction_raw, expanded_prior_probs], dim=4)
            concatenated_tensor = torch.cat([concatenated_tensor,expanded_action_contex], dim=4)
            final_tensor = torch.cat([concatenated_tensor, expanded_mse_loss], dim=4)

            expansion_values = final_tensor[:, :, :, 1, 0]
            action_values = final_tensor[:, :, 0, 0, 0].view(state_for_next_prior.shape[0], -1, 1)
            action_mse = final_tensor[:, :, 0, 0, -1]
            expansion_values *= (tree_gamma ** action_sequence)

            ## weighted means
            # expansion_mean = expansion_values.mean(dim=2)
            # action_mean = action_values.mean(dim=2)
            # mean_values = action_mean + 0.1 * (expansion_mean - action_mean)

            #average means
            mean_values = torch.cat((expansion_values, action_values), dim=2)
            mean_values = mean_values.mean(dim=2) -  mse_factor*action_mse
            mean_values = mean_values.mean(dim=2)

            k = max(int(mean_values.size(1) * action_percent), 1)
            all_selected_tensors = []
            values_with_b, index = torch.topk(mean_values, k)
            #start = time.time()
            #print(index.shape)
            #print()
            for i in range(state_for_next_prior.shape[0]):
                store_value(state_for_next_prior[i], final_tensor[i], index[i])
                all_selected_tensors.append(final_tensor[i][index[i]])


            final_selected_state = torch.cat(all_selected_tensors, dim=0)
            final_selected_state = final_selected_state.view(-1,final_selected_state.size(2), final_selected_state.size(3))
            #print(final_selected_tensor.shape, prediction_raw.shape)
            final_selected_state = final_selected_state[:,1,1:1+model.observation_dim]
            #print(state_for_next_prior.shape)

            state_for_next_prior = torch.unique(final_selected_state, dim=0)
            #print(state_for_next_prior.shape)
            #state_for_next_prior = prediction_raw[:,1,1:1+model.observation_dim]
            #state_for_next_prior = torch.unique(state_for_next_prior, dim=0)
    print("inference time,",time.time() - start)
    #mcts_instance = MCTS(state, state_dict, tree_gamma, prior, model, 1, 1, mse_factor, max_depth - 1)
    mcts_instance = MCTS(state, state_dict, tree_gamma, prior, model, int(n_action*action_percent), n_expand, mse_factor, max_depth-1)

    start_time = time.time()
    mcts_instance.search(mcts_itr)
    #print(mcts_instance.Qsa.values())
    values_list = list(mcts_instance.Qsa.values())

    # Stack the tensors into one tensor
    values_tensor = torch.stack(values_list)

    # Compute the mean and standard deviation
    value_mean = torch.mean(values_tensor)
    value_std = torch.std(values_tensor)
    value_max = torch.max(values_tensor)
    value_min = torch.min(values_tensor)
    # Print the results
    print("Mean:", value_mean.item(), "Std:", value_std.item(), "Max:", value_max.item(), "Min:", value_min.item())
    # Stop the timer
    end_time = time.time()

    # Calculate the running time
    running_time = end_time - start_time
    print("search time,", running_time)
    best_action = mcts_instance.best_action().long()
    print(best_action)
    prediction_raw = model.decode_from_indices(best_action.view(1, -1), state).squeeze(0)
    return prediction_raw.cpu().numpy()

@torch.no_grad()
def beam_with_prior_MTP(prior, model, x, denormalize_rew, denormalize_val, discount, steps,
                    beam_width, n_expand, n_action, b_percent, action_percent,
                    pw_alpha, mcts_itr, prob_threshold=0.05, likelihood_weight=5e2, prob_acc="product", return_info=False):
    contex = None
    #print(x.shape, prior.observation_dim)
    #print(x)
    state = x[:, 0, :prior.observation_dim]
    #print(state)
    acc_probs = torch.zeros([1]).to(x)
    acc_oods = torch.zeros([1]).to(x)
    info = {}
    values_track = None
    steps = 1
    for step in range(steps//model.latent_step):
        if step == 0:
            #logits, _ = prior(None, state) # [B x t x K]
            #start_time = time.time()
            logits, _ = prior(None, state)  # [B x t x K]
            #print(logits.shape)
            #prior_time = time.time() - start_time
            #print("prior function 1:", prior_time)
        else:
            contex = None
            logits, _ = prior(None, state_for_next_prior)
        #print("state shape:",state.unsqueeze(0))
        #logits = logits[:, -1, :]
        #print(logits.shape)
        #probs = torch.softmax(logits[:, :, -1, :], dim=-1)

        #probs = torch.softmax(logits[:, -1, :], dim=-1) # [B x K]
        #print(probs.shape)
        #print(probs)
        # = torch.log(probs)
        #nb_samples = 64 if step == 0 else n_expand
        head0_logits = logits[:, 0, -1, :]  # Shape: [B, vocab_size]
        probs = torch.softmax(head0_logits, dim=-1)  # Compute probabilities
        nb_samples = beam_width if step == 0 else n_expand
        samples = torch.multinomial(probs, num_samples=nb_samples, replacement=False) # [B, M]
        #print(samples.shape)
        contex = samples.reshape([-1, 1]) #[(B*M) x t]
        #print(contex.shape)
        #start_time = time.time()
        logits, _ = prior(contex, state)
        #prior_time = time.time() - start_time
        #print("prior function 2:", prior_time)
        #print(logits.shape)
        probs = torch.softmax(logits[:, :, -1, :], dim=-1)  # [B x K]
        #print(probs.shape)
        #log_probs = torch.log(probs)
        #samples = torch.multinomial(probs, num_samples=n_expand, replacement=True)  # [B, M]
        #contex = torch.cat([torch.repeat_interleave(contex, n_expand, 0), samples.reshape([-1, 1])], dim=1)
        #samples_log_prob = torch.cat([torch.index_select(a, 0, i).unsqueeze(0) for a, i in zip(log_probs, samples)]) # [B, M]


        #print(samples_log_prob)
        #state_for_next_prior = prediction_raw[:, 0, 1:model.observation_dim + 1]
        #print(contex.shape, contex)



        #if prob_acc in ["product", "expect"]:
        #    acc_probs = acc_probs.repeat_interleave(nb_samples*n_expand, 0) + samples_log_prob.reshape([-1])
        #start_time = time.time()
        prediction_raw = model.decode_from_indices(contex, state)
        #prior_time = time.time() - start_time
        #print("decode function:", prior_time)
        prediction = prediction_raw.reshape([-1, n_expand, 2, 3*model.action_dim+model.observation_dim+2])
        prediction_output = prediction[:, 0, :, :]
        V_t = prediction_raw[:, 1, 0]
        a_v = prediction_raw[:, 0, 0]
        if denormalize_val is not None:
            #V_t = denormalize_val(V_t).reshape([contex.shape[0], -1])
            V_t = denormalize_val(V_t)
            a_v = denormalize_val(a_v)
        #values = V_t[:, -1] * discounts[:, -1]
        #values = V_t[:, -1]
        #values = V_t * torch.exp(acc_probs)
        values = V_t
        values = values.reshape([-1, n_expand])
        a_v = a_v.view(beam_width, n_expand)

        # Select the first value from each group of 4 values
        a_v = a_v[:, 0]
        #print(a_v)
        # Reshape tensor_64 to [64, 1]
        a_v = a_v.view(beam_width, 1)

        # Concatenate along the second dimension (dim=1)
        result_tensor = torch.cat((values, a_v), dim=1)
        #print(result_tensor.shape)
        values_track = result_tensor.mean(dim=1)
        #print(result_tensor.shape)
        nb_top = beam_width if step < (steps // model.latent_step - 1) else 1

        #else:
        #    nb_top = 1
        #print(values_track.shape, nb_top)
        if prob_acc == "expect":
            values_with_b, index = torch.topk(values_track, nb_top)
            #print(index, result_tensor)
        else:
            values_with_b, index = torch.topk(values_track, nb_top)
        if return_info:
            info[(step+1)*model.latent_step] = dict(predictions=prediction_raw.cpu(), returns=values.cpu(),
                                                    latent_codes=contex.cpu(), log_probs=acc_probs.cpu(),
                                                    objectives=values+likelihood_bonus, index=index.cpu())
    optimal = prediction_output[index[0]]
    print(f"predicted max value {values_track[index[0]]}")
    if return_info:
        return optimal.cpu().numpy(), info
    else:
        return optimal.cpu().numpy()



@torch.no_grad()
def beam_with_uniform(prior, model, x, denormalize_rew, denormalize_val, discount, steps,
                    beam_width, n_expand,  prob_threshold=0.05):
    contex = None
    state = x[:, 0, :model.observation_dim]
    acc_probs = torch.ones([1]).to(x)
    for step in range(steps//model.latent_step):
        logits, _ = prior(contex, state) # [B x t x K]
        probs = torch.softmax(logits[:, -1, :], dim=-1) # [B x K]
        nb_samples = beam_width * n_expand if step == 0 else n_expand
        valid = probs > prob_threshold
        samples = torch.multinomial(valid/valid.sum(dim=-1), num_samples=nb_samples, replacement=True) # [B, M]
        samples_prob = torch.cat([torch.index_select(a, 0, i).unsqueeze(0) for a, i in zip(probs, samples)]) # [B, M]
        acc_probs = acc_probs.repeat_interleave(nb_samples, 0) * samples_prob.reshape([-1])
        if not contex is None:
            contex = torch.cat([torch.repeat_interleave(contex, nb_samples, 0), samples.reshape([-1, 1])],
                               dim=1)
        else:
            contex = samples.reshape([-1, step+1]) # [(B*M) x t]

        prediction_raw = model.decode_from_indices(contex, state)
        prediction = prediction_raw.reshape([-1, model.transition_dim])
        r_t, V_t = prediction[:, -3], prediction[:, -2]

        if denormalize_rew is not None:
            r_t = denormalize_rew(r_t).reshape([contex.shape[0], -1])
        if denormalize_val is not None:
            V_t = denormalize_val(V_t).reshape([contex.shape[0], -1])


        discounts = torch.cumprod(torch.ones_like(r_t) * discount, dim=-1)
        values = torch.sum(r_t[:,:-1] * discounts[:, :-1], dim=-1) + V_t[:,-1] * discounts[:,-1]
        nb_top = beam_width if step < (steps//model.latent_step-1) else 1
        values, index = torch.topk(values, nb_top)
        contex = contex[index]
        acc_probs = acc_probs[index]

    optimal = prediction_raw[index[0]]
    print(f"predicted max value {values[0]}")
    return optimal.cpu().numpy()

@torch.no_grad()
def beam_mimic(prior, model, x, denormalize_rew, denormalize_val, discount, steps,
                    beam_width, n_expand,  prob_threshold=0.05):
    contex = None
    state = x[:, 0, :model.observation_dim]
    acc_probs = torch.ones([1]).to(x)
    for step in range(steps//model.latent_step):
        logits, _ = prior(contex, state) # [B x t x K]
        probs = torch.softmax(logits[:, -1, :], dim=-1) # [B x K]
        nb_samples = beam_width * n_expand if step == 0 else n_expand
        samples = torch.multinomial(probs, num_samples=nb_samples, replacement=True) # [B, M]
        samples_prob = torch.cat([torch.index_select(a, 0, i).unsqueeze(0) for a, i in zip(probs, samples)]) # [B, M]
        acc_probs = acc_probs.repeat_interleave(nb_samples, 0) * samples_prob.reshape([-1])
        if not contex is None:
            contex = torch.cat([torch.repeat_interleave(contex, nb_samples, 0), samples.reshape([-1, 1])],
                               dim=1)
        else:
            contex = samples.reshape([-1, step+1]) # [(B*M) x t]

        nb_top = beam_width if step < (steps//model.latent_step-1) else 1
        values, index = torch.topk(acc_probs, nb_top)
        contex = contex[index]
        acc_probs = acc_probs[index]

    prediction_raw = model.decode_from_indices(contex, state)
    optimal = prediction_raw[0]
    print(f"value {values[0]}, prob {acc_probs[0]}")
    return optimal.cpu().numpy()


@torch.no_grad()
def enumerate_all(model, x, denormalize_rew, denormalize_val, discount):
    indicies = torch.range(0, model.model.K-1, device=x.device, dtype=torch.int32)
    prediction_raw = model.decode_from_indices(indicies, x[:, 0, :model.observation_dim])
    prediction = prediction_raw.reshape([-1, model.transition_dim])

    r_t, V_t = prediction[:, -2], prediction[:, -1]
    if denormalize_rew is not None:
        r_t = denormalize_rew(r_t).reshape([indicies.shape[0], -1])
    if denormalize_val is not None:
        V_t = denormalize_val(V_t).reshape([indicies.shape[0], -1])

    discounts = torch.cumprod(torch.ones_like(r_t) * discount, dim=-1)
    values = torch.sum(r_t[:,:-1] * discounts[:, :-1], dim=-1) + V_t[:,-1] * discounts[:,-1]
    optimal = prediction_raw[values.argmax()]
    return optimal.cpu().numpy()


@torch.no_grad()
def propose_plan_continuous(model, x):
    latent = torch.zeros([1, model.trajectory_embd], device="cuda")
    prediction = model.decode(latent, x[:, 0, :model.observation_dim])
    prediction = prediction.reshape([-1, model.transition_dim])
    return prediction.cpu().numpy()