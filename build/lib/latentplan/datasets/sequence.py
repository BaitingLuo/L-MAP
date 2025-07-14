import os
import numpy as np
import torch
import pdb

from latentplan.utils import discretization
from latentplan.utils.arrays import to_torch

from .d4rl import load_environment, qlearning_dataset_with_timeouts, minrl_dataset, qlearning_dataset_with_timeouts_training
from .preprocessing import dataset_preprocess_functions

def segment(observations, terminals, max_path_length):
    """
        segment `observations` into trajectories according to `terminals`
    """
    assert len(observations) == len(terminals)
    observation_dim = observations.shape[1]

    trajectories = [[]]
    for obs, term in zip(observations, terminals):
        trajectories[-1].append(obs)
        if term.squeeze():
            trajectories.append([])

    if len(trajectories[-1]) == 0:
        trajectories = trajectories[:-1]

    ## list of arrays because trajectories lengths will be different
    trajectories = [np.stack(traj, axis=0) for traj in trajectories]

    n_trajectories = len(trajectories)
    path_lengths = [len(traj) for traj in trajectories]

    ## pad trajectories to be of equal length
    trajectories_pad = np.zeros((n_trajectories, max_path_length, observation_dim), dtype=trajectories[0].dtype)
    early_termination = np.zeros((n_trajectories, max_path_length), dtype=np.bool)
    for i, traj in enumerate(trajectories):
        path_length = path_lengths[i]
        trajectories_pad[i,:path_length] = traj
        early_termination[i,path_length:] = 1
    return trajectories_pad, early_termination, path_lengths

class SequenceDataset(torch.utils.data.Dataset):
    def __init__(self, env, sequence_length=250, step=10, discount=0.99, max_path_length=1000,
                 penalty=None, device='cuda:0', normalize_raw=True, normalize_reward=True, train_portion=1.0, disable_goal=False, action_sequence_length=3):
        print(f'[ datasets/sequence ] Sequence length: {sequence_length} | Step: {step} | Max path length: {max_path_length}')
        self.env = env = load_environment(env) if type(env) is str else env
        self.sequence_length = sequence_length
        self.step = step
        self.max_path_length = max_path_length
        self.device = device
        self.disable_goal = disable_goal
        self.action_sequence_length = action_sequence_length
        print(f'[ datasets/sequence ] Loading...', end=' ', flush=True)
        if 'MineRL' in env.name:
            raise ValueError()
        #print("!!!!!!!!!!!!!!!!!!!", env.unwrapped)
        # dataset_for_normalization = qlearning_dataset_with_timeouts(env.unwrapped, terminate_on_end=True, disable_goal=disable_goal)
        # observations = dataset_for_normalization['observations']
        # actions = dataset_for_normalization['actions'].astype(np.float32)
        # rewards = dataset_for_normalization['rewards'].astype(np.float32)
        # #terminals = dataset_for_normalization['terminals']
        # #realterminals = dataset_for_normalization['realterminals']
        # print("observation shape,",observations.shape)
        # print("rewards shape,",rewards.shape)
        # self.normalized_raw = normalize_raw
        # self.normalize_reward = normalize_reward
        # self.obs_mean, self.obs_std = observations.mean(axis=0, keepdims=True), observations.std(axis=0, keepdims=True)
        # self.act_mean, self.act_std = actions.mean(axis=0, keepdims=True), actions.std(axis=0, keepdims=True)
        # self.reward_mean, self.reward_std = rewards.mean(), rewards.std()



        dataset = qlearning_dataset_with_timeouts(env.unwrapped, terminate_on_end=True, disable_goal=disable_goal)

        #dataset = qlearning_dataset_with_timeouts(env, dataset=None, terminate_on_end=False)
        print('✓')

        # preprocess_fn = dataset_preprocess_functions.get(env.name)
        # if preprocess_fn:
        #     print(f'[ datasets/sequence ] Modifying environment')
        #     dataset = preprocess_fn(dataset)
        #     print("!!!!!!!!!!!!!we are processing")
        ##
        observations = dataset['observations']
        actions = dataset['actions'].astype(np.float32)
        rewards = dataset['rewards'].astype(np.float32)
        terminals = dataset['terminals']
        realterminals = dataset['realterminals']
        print("observation shape,",observations.shape)
        print("rewards shape,",rewards.shape)

        self.normalized_raw = normalize_raw
        self.normalize_reward = normalize_reward
        self.obs_mean, self.obs_std = observations.mean(axis=0, keepdims=True), observations.std(axis=0, keepdims=True)
        self.act_mean, self.act_std = actions.mean(axis=0, keepdims=True), actions.std(axis=0, keepdims=True)
        self.reward_mean, self.reward_std = rewards.mean(), rewards.std()
        #self.normalized_raw = normalize_raw
        #self.normalize_reward = normalize_reward
        #self.obs_mean, self.obs_std = observations.mean(axis=0, keepdims=True), observations.std(axis=0, keepdims=True)
        #self.act_mean, self.act_std = actions.mean(axis=0, keepdims=True), actions.std(axis=0, keepdims=True)
        #self.reward_mean, self.reward_std = rewards.mean(), rewards.std()

        if normalize_raw:
            observations = (observations-self.obs_mean) / self.obs_std
            actions = (actions-self.act_mean) / self.act_std

        self.observations_raw = observations
        self.actions_raw = actions
        self.joined_raw = np.concatenate([observations, actions], axis=-1, dtype=np.float32)
        self.rewards_raw = rewards
        self.terminals_raw = terminals

        ## terminal penalty
        if penalty is not None:
            terminal_mask = realterminals.squeeze()
            self.rewards_raw[terminal_mask] = penalty
        #print(terminals)
        ## segment
        print(f'[ datasets/sequence ] Segmenting...', end=' ', flush=True)
        self.joined_segmented, self.termination_flags, self.path_lengths = segment(self.joined_raw, terminals, max_path_length)
        print(self.joined_segmented.shape)
        self.rewards_segmented, *_ = segment(self.rewards_raw, terminals, max_path_length)
        print('✓')

        self.discount = discount
        self.discounts = (discount ** np.arange(self.max_path_length))[:,None]

        ## [ n_paths x max_path_length x 1 ]
        self.values_segmented = np.zeros(self.rewards_segmented.shape, dtype=np.float32)

        for t in range(max_path_length):
            ## [ n_paths x 1 ]
            #change: we included the reward for current timestep
            V = (self.rewards_segmented[:, t:] * self.discounts[:max_path_length - t]).sum(axis=1)
            #V = (self.rewards_segmented[:,t+1:] * self.discounts[:-t-1]).sum(axis=1)
            self.values_segmented[:,t] = V
        #print(self.values_segmented[0])


        ## add (r, V) to `joined`
        values_raw = self.values_segmented.squeeze(axis=-1).reshape(-1)
        values_mask = ~self.termination_flags.reshape(-1)
        #print(values_mask)
        self.values_raw = values_raw[values_mask, None]

        if normalize_raw and normalize_reward:
            self.value_mean, self.value_std = self.values_raw.mean(), self.values_raw.std()
            self.values_raw = (self.values_raw-self.value_mean) / self.value_std
            #print("!!!!!!!!!!!!!!!!!Im checking value mean and std here:", np.mean(self.values_raw), np.std(self.values_raw))
            self.rewards_raw = (self.rewards_raw - self.reward_mean) / self.reward_std

            self.values_segmented = (self.values_segmented-self.value_mean) / self.value_std
            print("!!!!!!!!!!!!!!!!!Im checking value mean and std here:", np.mean(self.values_segmented),
                  np.std(self.values_segmented))
            self.rewards_segmented = (self.rewards_segmented - self.reward_mean) / self.reward_std
        else:
            self.value_mean, self.value_std = np.array(0), np.array(1)
        self.joined_raw = np.concatenate([self.joined_raw, self.rewards_raw, self.values_raw], axis=-1)
        self.joined_segmented = np.concatenate([self.joined_segmented, self.rewards_segmented, self.values_segmented], axis=-1)
        self.train_portion = train_portion
        self.test_portion = 1.0 - train_portion
        ## get valid indices
        indices = []
        test_indices = []
        print("self.path_lengths", len(self.path_lengths))
        for path_ind, length in enumerate(self.path_lengths):
            end = length - 1
            split = int(end * self.train_portion)
            for i in range(end):
                if i < split:
                    indices.append((path_ind, i, i+sequence_length))
                else:
                    test_indices.append((path_ind, i, i+sequence_length))
        print("self.indices", len(indices))
        self.indices = np.array(indices)
        self.test_indices = np.array(test_indices)
        self.observation_dim = observations.shape[1]
        self.action_dim = actions.shape[1]
        self.joined_dim = self.joined_raw.shape[1]

        ## pad trajectories
        n_trajectories, _, joined_dim = self.joined_segmented.shape
        self.joined_segmented = np.concatenate([
            self.joined_segmented,
            np.zeros((n_trajectories, sequence_length-1, joined_dim), dtype=np.float32),
        ], axis=1)

        #self.joined_segmented_tensor = torch.tensor(self.joined_segmented, device=device)
        self.termination_flags = np.concatenate([
            self.termination_flags,
            np.ones((n_trajectories, sequence_length-1), dtype=np.bool),
        ], axis=1)


    # def denormalize(self, states, actions, rewards, values):
    #     states = states*self.obs_std + self.obs_mean
    #     actions = actions*self.act_std + self.act_mean
    #     rewards = rewards*self.reward_std + self.reward_mean
    #     values = values*self.value_std + self.value_mean
    #     return states, actions, rewards, values
    #     return states, actions, values

    def denormalize(self, states, actions, values):
        states = states*self.obs_std + self.obs_mean
        actions = actions*self.act_std + self.act_mean
        values = values*self.value_std + self.value_mean
        return states, actions, values

    # def normalize_joined_single(self, joined):
    #     joined_std = np.concatenate([self.obs_std[0], self.act_std[0], self.reward_std[None], self.value_std[None]])
    #     joined_mean = np.concatenate([self.obs_mean[0], self.act_mean[0], self.reward_mean[None], self.value_mean[None]])
    #     return (joined-joined_mean) / joined_std

    def normalize_joined_single(self, joined):
        joined_std = np.concatenate([self.value_std[None], self.obs_std[0], self.act_std[0], self.act_std[0], self.act_std[0]])
        joined_mean = np.concatenate([self.value_mean[None], self.obs_mean[0], self.act_mean[0], self.act_mean[0], self.act_mean[0]])
        return (joined-joined_mean) / joined_std

    # def denormalize_joined(self, joined):
    #     states = joined[:,:self.observation_dim]
    #     actions = joined[:,self.observation_dim:self.observation_dim+self.action_dim]
    #     rewards = joined[:,-3, None]
    #     values = joined[:,-2, None]
    #     results = self.denormalize(states, actions, rewards, values)
    #     return np.concatenate(results+(joined[:, -1, None],), axis=-1)

    def denormalize_joined(self, joined):
        states = joined[:,1:self.observation_dim+1]
        actions = joined[:,1+self.observation_dim:self.observation_dim+1+self.action_dim]
        #rewards = joined[:,-3, None]
        values = joined[:,0, None]
        results = self.denormalize(states, actions, values)
        return np.concatenate(results+(joined[:, -1, None],), axis=-1)

    def normalize_states(self, states):
        if torch.is_tensor(states):
            obs_std = torch.Tensor(self.obs_std).to(states.device)
            obs_mean = torch.Tensor(self.obs_mean).to(states.device)
        else:
            obs_std = np.squeeze(np.array(self.obs_std))
            obs_mean =  np.squeeze(np.array(self.obs_mean))
        states = (states - obs_mean) / obs_std
        return states

    def denormalize_states(self, states):
        if torch.is_tensor(states):
            act_std = torch.Tensor(self.obs_std).to(states.device)
            act_mean = torch.Tensor(self.obs_mean).to(states.device)
        else:
            act_std = np.squeeze(np.array(self.obs_std))
            act_mean = np.squeeze(np.array(self.obs_mean))
        states = states * act_std + act_mean
        return states

    def denormalize_actions(self, actions):
        if torch.is_tensor(actions):
            act_std = torch.Tensor(self.act_std).to(actions.device)
            act_mean = torch.Tensor(self.act_mean).to(actions.device)
        else:
            act_std = np.squeeze(np.array(self.act_std))
            act_mean =  np.squeeze(np.array(self.act_mean))
        actions = actions*act_std + act_mean
        return actions

    def normalize_actions(self, actions):
        if torch.is_tensor(actions):
            act_std = torch.Tensor(self.act_std).to(actions.device)
            act_mean = torch.Tensor(self.act_mean).to(actions.device)
        else:
            act_std = np.squeeze(np.array(self.act_std))
            act_mean =  np.squeeze(np.array(self.act_mean))
        actions = (actions - act_mean) / act_std
        return actions

    def denormalize_rewards(self, rewards):
        if (not self.normalized_raw) or (not self.normalize_reward):
            return rewards
        if torch.is_tensor(rewards):
            reward_std = torch.Tensor([self.reward_std]).to(rewards.device)
            reward_mean = torch.Tensor([self.reward_mean]).to(rewards.device)
        else:
            reward_std = np.array([self.reward_std])
            reward_mean =  np.array([self.reward_mean])
        rewards = rewards*reward_std + reward_mean
        return rewards

    def denormalize_values(self, values):
        if (not self.normalized_raw) or (not self.normalize_reward):
                return values
        if torch.is_tensor(values):
            value_std = torch.Tensor([self.value_std]).to(values.device)
            value_mean = torch.Tensor([self.value_mean]).to(values.device)
        else:
            value_std = np.array([self.value_std])
            value_mean =  np.array([self.value_mean])
        values = values*value_std + value_mean
        return values

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        # path_ind, start_ind, end_ind = self.indices[idx]
        # #print(self.joined_segmented.shape)
        # #print(path_ind, start_ind, end_ind)
        # joined = self.joined_segmented[path_ind, start_ind:end_ind:self.step]
        # print("join shape:",joined.shape)
        # ## don't compute loss for parts of the prediction that extend
        # ## beyond the max path length
        # traj_inds = torch.arange(start_ind, end_ind, self.step)
        # mask = torch.ones(joined.shape, dtype=torch.bool)
        # mask[traj_inds > self.max_path_length - self.step] = 0
        # terminal = 1-torch.cumprod(~torch.tensor(self.termination_flags[path_ind, start_ind:end_ind:self.step, None]),
        #                           dim=1)
        # ## flatten everything
        # X = joined[:-1]
        # Y = joined[1:]
        # #print(X.shape)
        # mask = mask[:-1]
        # terminal = terminal[:-1]
        ################################################################
        path_ind, start_ind, end_ind = self.indices[idx]
        joined = self.joined_segmented[path_ind, start_ind:end_ind:self.step]

        state_dim = self.observation_dim
        action_dim = self.action_dim

        # Extract components
        states = joined[:, :state_dim]
        actions = joined[:, state_dim:state_dim + action_dim]
        rewards = joined[:, -1]  # Assuming value are last column
        # Calculate how many full sequences we can form
        total_length = len(joined) // self.action_sequence_length
        #print("Data type of 'rewards':", rewards.dtype)

        # Define dimensions for X and Y arrays
        X = np.zeros((total_length, 1 + state_dim + action_dim * self.action_sequence_length))
        Y = np.zeros_like(X)
        #print("Data type of 'X':", X.dtype)

        # Compute mask based on max path length
        #print(rewards,total_length,X.shape)
        for i in range(total_length):
            base_index = i * self.action_sequence_length
            if base_index + self.action_sequence_length > len(joined):
                break  # Break if there aren't enough actions to form a full sequence
            #print(base_index)
            X[i, 0] = rewards[base_index]  # Reward
            X[i, 1:1 + state_dim] = states[base_index]  # State
            for j in range(self.action_sequence_length):
                action_index = base_index + j
                X[i, 1 + state_dim + j * action_dim:1 + state_dim + (j + 1) * action_dim] = actions[action_index]

        # Define Y based on the next sequence
        if total_length > 1:
            Y[:-1] = X[1:]
        traj_inds = torch.arange(start_ind, end_ind, self.step)
        mask = torch.ones((joined.shape[0], X.shape[1]), dtype=torch.bool)
        mask[traj_inds > self.max_path_length - self.step] = 0
        terminal = 1-torch.cumprod(~torch.tensor(self.termination_flags[path_ind, start_ind:end_ind:self.step, None]), dim=1)
        # Adjust mask and terminal to match the sequences
        mask = mask[::self.action_sequence_length][:total_length]
        terminal = terminal[::self.action_sequence_length][:total_length]
        #print(X.shape, Y.shape, mask.shape, terminal.shape)
        return X, Y, mask, terminal

    def get_test(self):
        Xs = []
        Ys = []
        masks = []
        terminals = []
        for path_ind, start_ind, end_ind in self.test_indices:
            joined = self.joined_segmented[path_ind, start_ind:end_ind:self.step]

            ## don't compute loss for parts of the prediction that extend
            ## beyond the max path length
            traj_inds = torch.arange(start_ind, end_ind, self.step)
            mask = torch.ones(joined.shape, dtype=torch.bool)
            mask[traj_inds > self.max_path_length - self.step] = 0
            terminal = 1 - torch.cumprod(
                ~torch.tensor(self.termination_flags[path_ind, start_ind:end_ind:self.step, None]),
                dim=1)

            ## flatten everything
            X = joined[:-1]
            Y = joined[1:]
            mask = mask[:-1]
            terminal = terminal[:-1]
            Xs.append(torch.tensor(X))
            Ys.append(torch.tensor(Y))
            masks.append(torch.tensor(mask))
            terminals.append(torch.tensor(terminal))
        return torch.stack(Xs), torch.stack(Ys), torch.stack(masks), torch.stack(terminals)

def one_hot(a, num_classes):
  return np.squeeze(np.eye(num_classes, dtype=np.uint8)[a.reshape(-1)])