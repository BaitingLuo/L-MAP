from collections import defaultdict
import torch
import time

REWARD_DIM = VALUE_DIM = 1
from .mcts_expand import *
import networkx as nx
import torch.nn.functional as F

@torch.no_grad()
def MCTS_P(prior, model, x, initial_width, n_expand, n_action, b_percent, action_percent,
                    pw_alpha, mcts_itr, macro_step=3, depth=3):
    state = x[:, 0, :prior.observation_dim]
    def tensor_to_tuple(tensor):
        return tuple(tensor.cpu().numpy().flatten())

    # Initialize the cache/ Pre-constructed search space
    state_dict = {}
    # Store the value in the nested dictionary
    def store_value(state, action_matrix, index):
        state_key = tensor_to_tuple(state)
        if state_key not in state_dict:
            state_dict[state_key] = [action_matrix,index]
    import time
    start = time.time()
    max_depth = depth
    tree_gamma = 0.99
    action_sequence = macro_step
    mse_factor = 0
    for step in range(max_depth):
        if step == 0:
            logits, _ = prior(None, state) # [B x t x K]
        else:
            #contex = None
            logits, _ = prior(None, state_for_next_prior)
        action_probs = torch.softmax(logits[:, -1, :], dim=-1) # [B x K]
        nb_samples = initial_width if step == 0 else n_action
        action_samples = torch.multinomial(action_probs, num_samples=nb_samples, replacement=False) # [B, M]
        # Gather the corresponding probabilities for the sampled actions
        action_probs_sampled = torch.gather(action_probs, 1, action_samples)
        action_contex = action_samples.reshape([-1, 1]) # [(B*M) x t]
        if step == 0:
            logits, _ = prior(action_contex, state)
        else:
            state_for_next_prior_expanded = state_for_next_prior.repeat_interleave(nb_samples, 0)
            logits, _ = prior(action_contex, state_for_next_prior_expanded)
        probs = torch.softmax(logits[:, -1, :], dim=-1)  # [B x K]
        log_probs = torch.log(probs)
        samples = torch.multinomial(probs, num_samples=n_expand, replacement=True)  # [B, M]
        contex = torch.cat([torch.repeat_interleave(action_contex, n_expand, 0), samples.reshape([-1, 1])], dim=1)
        if step == 0:
            prediction_raw = model.decode_from_indices(contex, state)
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
            expansion_values = final_tensor[:, :, 1, 0]   #return to go for sampled state
            action_values = final_tensor[:, 0, 0, 0].view(-1, 1)  #return to go for bootstrapping Q value
            action_mse = final_tensor[:, 0, 0, -1]
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

            reshaped_prediction_raw = prediction_raw.view(-1,nb_samples, n_expand, 2, model.observation_dim + action_sequence*model.action_dim+ 2)
            action_contex = action_contex.view(-1, nb_samples, 1)
            action_probs_sampled = action_probs_sampled.view(-1, nb_samples, 1)
            expanded_prior_probs = action_probs_sampled.unsqueeze(3).unsqueeze(4).expand(-1, nb_samples,
                                                                                             n_expand, 2, 1)

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

            k = max(int(mean_values.size(1) * action_percent), 1)
            all_selected_tensors = []
            values_with_b, index = torch.topk(mean_values, k)
            #start = time.time()
            for i in range(state_for_next_prior.shape[0]):
                store_value(state_for_next_prior[i], final_tensor[i], index[i])
                all_selected_tensors.append(final_tensor[i][index[i]])


            final_selected_state = torch.cat(all_selected_tensors, dim=0)
            final_selected_state = final_selected_state.view(-1,final_selected_state.size(2), final_selected_state.size(3))
            final_selected_state = final_selected_state[:,1,1:1+model.observation_dim]

            state_for_next_prior = torch.unique(final_selected_state, dim=0)
    print("inference time,",time.time() - start)
    mcts_instance = MCTS(state, state_dict, tree_gamma, prior, model, int(n_action*action_percent), n_expand, mse_factor, max_depth-1, pw_alpha)

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
def MCTS_F(prior, model, x, initial_width, n_expand, n_action, b_percent, action_percent,
                    pw_alpha, mcts_itr, macro_step=3, depth=3):
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
    max_depth = depth
    tree_gamma = 0.99
    action_sequence = macro_step
    mse_factor = 0
    for step in range(max_depth):
        if step == 0:
            logits, _ = prior(None, state) # [B x t x K]
        else:
            logits, _ = prior(None, state_for_next_prior) #used to sample intermediate action contex
        action_probs = torch.softmax(logits[:, -1, :], dim=-1) # [B x K]
        nb_samples = initial_width if step == 0 else n_action

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
            reshaped_prediction_raw = prediction_raw.view(-1,nb_samples, n_expand, 2, model.observation_dim + action_sequence*model.action_dim+ 2)
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
    mcts_instance = MCTS(state, state_dict, tree_gamma, prior, model, int(n_action*action_percent), n_expand, mse_factor, max_depth-1, pw_alpha)

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