import math
import random
#from nsbridge_simulator.nsbridge_v0 import NSBridgeV0 as model
#from nsfrozenlake.nsfrozenlake_v0 import NSFrozenLakeV0 as model
import pickle
#from BayesianNeuralNetwork import *
import numpy as np
#import utils.distribution as distribution
from multiprocessing import Pool
import sys
import torch
import time
import random
import time
import cProfile
import torch.nn.functional as F


#from latentplan.search import enumerate_all

# Set a higher recursion depth limit (e.g., 3000)
sys.setrecursionlimit(100000)


def tensor_to_tuple(tensor):
    return tuple(tensor.cpu().numpy().flatten())

class Cache:
    def __init__(self):
        self.model_cache = {}
        self.prior_cache = {}
        self.decoder_cache = {}
    def get_decoder_cache(self, key):
        return self.decoder_cache.get(key, None)

    def set_decoder_cache(self, key, value):
        self.decoder_cache[key] = value

    def get_model_cache(self, key):
        return self.model_cache.get(key, None)

    def set_model_cache(self, key, value):
        self.model_cache[key] = value

    def get_prior_cache(self, key):
        return self.prior_cache.get(key, None)

    def set_prior_cache(self, key, value):
        self.prior_cache[key] = value
class Node:

    def __init__(self, state, tree_gamma, prior, depth, result_value='undecided', prior_prob = None, expanded = False, transition = None, action=None, parent=None, node_type="decision", mcts=None, ood_value=None, resamples=None):
        self.state = state
        self.action = action  # Action taken or outcome for chance node
        self.transition = transition
        self.parent = parent
        self.children = []
        self.visits = 0
        self.value = 0.0
        self.type = node_type  # "decision" or "chance"
        self.probabilities = [] if self.type == "chance" else None
        #self.possible_actions = [0, 1, 2, 3]
        self.tree_gamma = tree_gamma
        #self.time = time
        self.np_random = np.random.RandomState()
        self.mcts = mcts  # Add reference to the MCTS instance
        self.prior = prior
        #self.contex = contex
        self.depth = depth
        self.model = mcts.model
        #self.OOD = OOD_list
        self.return_to_go = result_value
        self.ood_value = ood_value
        self.resamples = resamples
        self.expanded = expanded
        self.node_value = None
        self.prior_prob = prior_prob
    def is_decision_node(self):
        return self.type == "decision"

    def is_chance_node(self):
        return self.type == "chance"

    def get_child_with_state(self, sampled_state):
        for child in self.children:
            # Use torch.equal to check if two tensors are equal
            #if torch.equal(child.contex, contex):
            if child.state == sampled_state:
                return child
        return None

    def store_value(self, state, action_matrix, index):
        state_key = tensor_to_tuple(state)
        if state_key not in self.mcts.state_dict:
            self.mcts.state_dict[state_key] = [action_matrix, index]

    def expand(self):
        child_node = None
        if self.is_decision_node():
            if self.mcts.is_terminal(self.depth) and self.node_value is not None:
                return self
            retrieved_data = self.mcts.state_dict.get(self.state, None)
            if retrieved_data is not None:
                actions_samples = retrieved_data[0]
                expanded_actions = retrieved_data[1]
                actions_samples = actions_samples[expanded_actions]
                all_actions = actions_samples[:,0,0,-2].flatten()
                expansion_values = actions_samples[:,:,1,0]
                action_values = actions_samples[:,0,0,0].view(-1,1)
                expanded_states = actions_samples[:,:,1,1:self.mcts.model.observation_dim+1]
                #indicate how good the action reconstruct the first state, OOD indicator
                action_mse = actions_samples[:, 0, 0, -1]
                expansion_values *= (self.tree_gamma ** self.mcts.action_sequence)
                mean_values = torch.cat((expansion_values, action_values), dim=1)
                self.increase_visit = mean_values.numel()
                expansion_factor = mean_values.size(1)
                mean_values = mean_values.mean(dim=1)

                mean_values = mean_values - self.mcts.mse_factor*action_mse
                self.node_value = mean_values.mean(dim=0)
                prior_prob = actions_samples[:, 0, 0, -3]
                self.proposed_prob = 0
                for itr, action in enumerate(all_actions):
                    if itr in expanded_actions:
                        chosen_action = True
                    else:
                        chosen_action = False
                    self.proposed_prob += prior_prob[itr]
                    child_node = Node(self.state, self.tree_gamma, self.prior, self.depth, prior_prob = prior_prob[itr], expanded=chosen_action, action=action, parent=self, node_type="chance", mcts=self.mcts, ood_value=action_mse[itr], resamples=expanded_states[itr])
                    self.children.append(child_node)
                    #self.mcts.update_metrics(self.state, action, (self.tree_gamma ** self.depth) * mean_values[itr], expansion_factor)
                    self.mcts.update_metrics(self.state, action, mean_values[itr], expansion_factor)
            else:
                #get into a new state, needs to do expansion online
                input_state = torch.tensor(self.state).reshape([1, -1]).to('cuda')
                logits, _ = self.mcts.prior(None, input_state)
                action_probs = torch.softmax(logits[:, -1, :], dim=-1)  # [B x K]
                action_samples = torch.multinomial(action_probs, num_samples=self.mcts.n_action, replacement=False)  # [B, M]
                action_contex = action_samples.reshape([-1, 1])  # [(B*M) x t]
                action_probs_sampled = torch.gather(action_probs, 1, action_samples)
                logits, _ = self.mcts.prior(action_contex, input_state)
                probs = torch.softmax(logits[:, -1, :], dim=-1)  # [B x K]
                samples = torch.multinomial(probs, num_samples=self.mcts.n_expand, replacement=True)  # [B, M]
                contex = torch.cat([torch.repeat_interleave(action_contex, self.mcts.n_expand, 0), samples.reshape([-1, 1])],
                                   dim=1)
                prediction_raw = self.mcts.model.decode_from_indices(contex, input_state)
                reshaped_prediction_raw = prediction_raw.view(self.mcts.n_action, self.mcts.n_expand, 2, -1)
                expanded_action_contex = action_contex.unsqueeze(1).unsqueeze(2).expand(self.mcts.n_action, self.mcts.n_expand, 2, 1)
                predicted_first_state = prediction_raw[:, 0, 1:1 + self.mcts.model.observation_dim]
                decoded_state_compare = input_state.expand_as(predicted_first_state)
                mse_loss_per_element = F.mse_loss(predicted_first_state, decoded_state_compare, reduction='none')
                mse_loss_per_example = mse_loss_per_element.mean(dim=1)
                mse_loss_per_example = mse_loss_per_example.view(self.mcts.n_action, self.mcts.n_expand)
                expanded_mse_loss = mse_loss_per_example.unsqueeze(2).unsqueeze(3).expand(self.mcts.n_action, self.mcts.n_expand, 2, 1)
                expanded_prior_probs = action_probs_sampled.reshape([-1, 1]).unsqueeze(2).unsqueeze(3).expand(
                    self.mcts.n_action, self.mcts.n_expand, 2, 1)
                concatenated_tensor = torch.cat([reshaped_prediction_raw, expanded_prior_probs], dim=3)
                concatenated_tensor = torch.cat([concatenated_tensor, expanded_action_contex], dim=3)
                final_tensor = torch.cat([concatenated_tensor, expanded_mse_loss], dim=3)
                expansion_values = final_tensor[:, :, 1, 0]
                action_values = final_tensor[:, 0, 0, 0].view(-1, 1)
                action_mse = final_tensor[:, 0, 0, -1]
                expansion_values *= (self.tree_gamma ** self.mcts.action_sequence)
                mean_values = torch.cat((expansion_values, action_values), dim=1)
                self.increase_visit = mean_values.numel()
                expansion_factor = mean_values.size(1)
                mean_values = mean_values.mean(dim=1)
                mean_values = mean_values - self.mcts.mse_factor*action_mse
                self.node_value = mean_values.mean(dim=0)
                k = int(mean_values.size(0))
                values_with_b, index = torch.topk(mean_values, k)
                self.store_value(input_state, final_tensor, index)
                action_probs_sampled = action_probs_sampled.view(-1)
                expanded_states = reshaped_prediction_raw[:, :, 1, 1:self.mcts.model.observation_dim + 1]
                self.proposed_prob = 0
                for itr, action in enumerate(action_contex.flatten()):
                    self.proposed_prob += action_probs_sampled[itr]
                    child_node = Node(self.state, self.tree_gamma, self.prior, self.depth, prior_prob = action_probs_sampled[itr], expanded=True, action=action, parent=self, node_type="chance", mcts=self.mcts, ood_value=action_mse[itr], resamples=expanded_states[itr])
                    self.children.append(child_node)
                    self.mcts.update_metrics(self.state, action, mean_values[itr], expansion_factor)
        else:  # For a chance node
            #let's do random sampling for now
            sa = (self.state, self.action)
            visit_count = self.mcts.Nsa.get(sa, 0)
            k = 1
            alpha = 0
            #Progressive widening. k is normally set to 1. Alpha is used for controlling propensity, set to 0 for improving efficiency.
            if len(self.parent.children) < k * (visit_count ** alpha):
                action_contex = self.action.long().reshape([-1, 1])
                input_state = torch.tensor(self.state).reshape([1, -1]).to('cuda')
                logits, _ = self.mcts.prior(action_contex, input_state)
                probs = torch.softmax(logits[:, -1, :], dim=-1)  # [B x K]
                samples = torch.multinomial(probs, num_samples=1, replacement=True)  # [B, M]
                contex = torch.cat([torch.repeat_interleave(action_contex, 1, 0), samples.reshape([-1, 1])], dim=1)
                prediction_raw = self.mcts.model.decode_from_indices(contex, input_state)
                predicted_first_state = prediction_raw[:, 0, 1:1 + self.mcts.model.observation_dim]
                sampled_state = tensor_to_tuple(predicted_first_state)
                existing_child = self.get_child_with_state(sampled_state)
                if existing_child is not None:
                    child_node = existing_child
                else:
                    child_node = Node(sampled_state, self.tree_gamma, self.prior, self.depth+1, parent=self, node_type="decision", mcts=self.mcts)
                    self.children.append(child_node)
            else:
                num_samples = 1
                sampled_indice = torch.randint(0, self.resamples.size(0), (num_samples,))
                sampled_tensor = self.resamples[sampled_indice]
                sampled_state = tensor_to_tuple(sampled_tensor)
                # Check if a child with the resulting state already exists
                existing_child = self.get_child_with_state(sampled_state)
                if existing_child is not None:
                    child_node = existing_child
                else:
                    child_node = Node(sampled_state, self.tree_gamma, self.prior, self.depth+1, parent=self, node_type="decision", mcts=self.mcts)
                    self.children.append(child_node)
            return child_node

        return self

    def get_return(self):
        #if len(self.mcts.context_to_next_tokens[self.depth][tuple(self.contex.cpu().numpy().flatten())]['values']) == 0:
        #    print(self.depth, tuple(self.contex.cpu().numpy().flatten()))
        #    raise MyError(
        #        f"Empty values encountered at depth {self.depth} with context {tuple(self.contex.cpu().numpy().flatten())}.")
        #if self.depth >= 4:
        #    print("depth:", self.depth, self.contex,self.mcts.context_to_next_tokens[self.depth][tuple(self.contex.cpu().numpy().flatten())]['values'][0])
        #return self.mcts.context_to_next_tokens[self.depth][tuple(self.contex.cpu().numpy().flatten())]['values'][0]
        return self.return_to_go


    def backpropagate(self, node_value, increase_visit, depth):
        #self.visits += 1
        if self.parent:
            if self.is_chance_node():
                depth_count = depth + self.mcts.action_sequence
            else:
                depth_count = depth
            self.parent.backpropagate(node_value, increase_visit, depth_count)
            if self.action is not None:  # Ensure the action is valid (not None)
                # Note: Discounting applied here might need adjustment based on how you want to use it in metric updates
                self.mcts.update_metrics(self.state, self.action,
                                         (self.tree_gamma ** depth_count) * node_value,
                                         increase_visit)

    def best_child(self, exploration_constant=math.sqrt(2)):
        best_value = float('-inf')
        best_nodes = []
        s = self.state
        for child in self.children:
            #start = time.time()
            action = child.action
            sa = (s, action)
            #print(sa)
            if sa in self.mcts.Qsa:
                #print(self.mcts.Nsa[sa])
                ucb_value = self.mcts.Qsa[sa] + exploration_constant * math.sqrt(
                    math.log(self.mcts.Ns.get(s, 1)) / self.mcts.Nsa[sa])
            else:
                #ucb_value = exploration_constant * math.sqrt(
                #    math.log(self.mcts.Ns.get(self.state, 1)) / 1)  # Assume at least one visit
                ucb_value = float('inf')
            if ucb_value > best_value:
                best_value = ucb_value
                best_nodes = [child]
            elif ucb_value == best_value:
                best_nodes.append(child)
            #print("itr time", time.time() - start)
        return random.choice(best_nodes) if best_nodes else None

    def p_best_child(self, exploration_constant=math.sqrt(2)):
        best_value = float('-inf')
        best_nodes = []
        s = self.state
        for child in self.children:
            #start = time.time()
            action = child.action
            sa = (s, action)
            #print(sa)
            if sa in self.mcts.Qsa:
                #print(self.mcts.Nsa[sa])
                ucb_value = self.mcts.Qsa[sa] + exploration_constant * child.prior_prob * math.sqrt(
                    math.log(self.mcts.Ns.get(s, 1)) / self.mcts.Nsa[sa])
            else:
                #ucb_value = exploration_constant * math.sqrt(
                #    math.log(self.mcts.Ns.get(self.state, 1)) / 1)  # Assume at least one visit
                ucb_value = float('inf')
            if ucb_value > best_value:
                best_value = ucb_value
                best_nodes = [child]
            elif ucb_value == best_value:
                best_nodes.append(child)
            #print("itr time", time.time() - start)
        return random.choice(best_nodes) if best_nodes else None

    def p_best_child2(self, exploration_constant=math.sqrt(2)):
        best_value = float('-inf')
        best_nodes = []
        s = self.state
        for child in self.children:
            #start = time.time()
            action = child.action
            sa = (s, action)
            if sa in self.mcts.Qsa:
                #print(exploration_constant * (self.mcts.Nsa[sa]/self.mcts.Ns.get(s, 1)*child.prior_prob/self.proposed_prob))
                # ucb_value = self.mcts.Qsa[sa] + exploration_constant * (1/len(self.children)) * child.prior_prob * math.sqrt(
                #     math.log(self.mcts.Ns.get(s, 1)) / self.mcts.Nsa[sa])
                # π(s,a) is child.prior_prob (the prior policy)
                prior_prob = child.prior_prob.cpu()  # This is the single action prior probability \pi(s,a)

                # Generate noise for a single action (you can use normal or uniform noise)
                epsilon = 0.25  # Example noise factor, adjust based on paper's recommendation
                noise = torch.rand(1).item()  # Generates a random value between 0 and 1 (uniform noise)

                # Mix the prior with the noise to get the noisy proposal probability β(s,a)
                proposal_prob = (1 - epsilon) * prior_prob + epsilon * noise  # β(s,a)

                # Correction factor: Empirical distribution over proposal distribution
                empirical_prob = 1 / len(self.children)  # Assuming empirical_prob is uniformly distributed
                correction_factor = empirical_prob / proposal_prob

                # Apply the correction factor to the prior probability
                corrected_prior_prob = correction_factor * prior_prob

                # Convert corrected_prior_prob to a scalar (if needed)
                corrected_prior_prob_scalar = corrected_prior_prob.item()

                # Now use corrected_prior_prob_scalar in the UCT formula
                ucb_value = self.mcts.Qsa[sa] + exploration_constant * corrected_prior_prob_scalar * math.sqrt(
                    math.log(self.mcts.Ns.get(s, 1)) / self.mcts.Nsa[sa])
            else:
                #ucb_value = exploration_constant * math.sqrt(
                #    math.log(self.mcts.Ns.get(self.state, 1)) / 1)  # Assume at least one visit
                ucb_value = float('inf')
            if ucb_value > best_value:
                best_value = ucb_value
                best_nodes = [child]
            elif ucb_value == best_value:
                best_nodes.append(child)
            #print("itr time", time.time() - start)
        return random.choice(best_nodes) if best_nodes else None

class MCTS:
    def __init__(self, state, state_dict, tree_gamma, prior, model, n_action, n_expand, mse_factor, max_depth):
        #self.root = Node(bnn1.task._NSBridgeV0__decode_state(initial_state_coordinate, initial_state_index, -1))
        initial_state = 'root'
        contex_state = None
        contex = None
        self.state_dict = state_dict
        self.OOD_list = []
        self.cache = Cache()
        self.prior = prior
        self.model = model
        self.root = Node(tensor_to_tuple(state), tree_gamma, prior, 0, mcts=self)
        # Metrics
        self.Qsa = {}  # stores Q values for s,a pairs
        self.Nsa = {}  # stores visit counts for s,a pairs
        self.Ns = {}   # stores visit counts for states
        self.unadded = []
        self.max_depth = max_depth
        self.n_action = n_action
        self.n_expand = n_expand
        self.action_sequence = 3
        self.mse_factor = mse_factor


    def is_terminal(self, depth):
        #reward = self.task.instant_reward_byindex(state)
        if depth == self.max_depth:
            return True

    def search(self, iterations):
        for _ in range(iterations):
            leaf = self.traverse(self.root)  # Traverse till you reach a leaf
            expanded_node = leaf.expand()
            expanded_node.backpropagate(expanded_node.node_value,expanded_node.increase_visit,0)
    def traverse(self, node):
        while node.children:
            #if node.is_decision_node() and self.is_terminal(node.depth):
            #    return node
            if self.is_terminal(node.depth):
                return node
            if node.is_decision_node():
                selected_chance_node = node.best_child()
                #selected_chance_node = node.p_best_child()
                node = selected_chance_node.expand()
        return node

    def best_action(self):
        best_avg_value = -float('inf')
        best_q_value = -float('inf')
        best_action = None
        #print(self.Nsa)
        for child in self.root.children:
            sa = (child.state, child.action)  # Create a state-action pair

            # Fetch Q-value and visit count in a single lookup to avoid redundant dictionary accesses
            q_value = self.Qsa.get(sa, None)
            visit_count = self.Nsa.get(sa, 0)
            #print(q_value, visit_count)
            # If the state-action pair has been visited at least once
            if q_value is not None and visit_count > 0:
                # Check if this action has a better visit count or, in the case of a tie, a better Q-value
                if (visit_count > best_avg_value) or (visit_count == best_avg_value and q_value > best_q_value):
                    best_avg_value = visit_count
                    best_q_value = q_value
                    best_action = child.action
        return best_action

    def update_metrics(self, state, action, reward, expansion_factor):
        state_key = state
        action_key = action

        sa = (state_key, action_key)  # Use the converted state and action as the key
        if sa in self.Qsa:
            self.Qsa[sa] = (self.Qsa[sa] * self.Nsa[sa] + reward * expansion_factor) / (self.Nsa[sa] + expansion_factor)
            self.Nsa[sa] += expansion_factor
        else:
            self.Qsa[sa] = reward
            self.Nsa[sa] = expansion_factor

        # Update Ns for the state
        if state_key in self.Ns:
            self.Ns[state_key] += expansion_factor
        else:
            self.Ns[state_key] = expansion_factor
