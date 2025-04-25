"""
Trajectory data collection utilities for reinforcement learning.

This module provides tools for collecting trajectory data from trained RL policies
in D4RL-compatible format for offline reinforcement learning research.
"""


"""
    Standard Libraries
"""
import os
from typing import Dict, Optional, Tuple

"""
    3rd Party Libraries
"""
import gymnasium as gym
import minari
from minari import DataCollector, StepDataCallback
import numpy as np
from stable_baselines3 import SAC
from stable_baselines3.common.evaluation import evaluate_policy
import torch


class TypeConsistentCallback(StepDataCallback):
    """
    A strict callback that ensures completely consistent data types.
    Prevents PyArrow errors by guaranteeing type consistency.
    """
    def __init__(self):
        # Store the initial observation shape once we see it
        self.obs_shape = None
        # Keep a set of required info keys for each environment type
        self.env_type_keys = {
            "HalfCheetah": ['reward_ctrl', 'reward_run', 'x_position', 'x_velocity']
        }
        # Whether we've seen the first step yet
        self.initialized = False
        # Store the initially seen info keys 
        self.info_keys = None
        
    def __call__(self, env=None, obs=None, action=None, next_obs=None, reward=None, 
                terminated=None, truncated=None, info=None, **kwargs):
        """
        Create step data with strictly consistent types and shapes.
        """
        # Determine environment type
        env_id = "unknown"
        if env and hasattr(env, 'unwrapped') and hasattr(env.unwrapped, 'spec'):
            if hasattr(env.unwrapped.spec, 'id'):
                env_id = env.unwrapped.spec.id
        
        # Get the environment base type (e.g., "HalfCheetah" from "HalfCheetah-v4")
        env_base = next((et for et in self.env_type_keys if et in env_id), None)
        
        # If this is the first call, initialize the observation shape
        if not self.initialized and obs is not None:
            self.obs_shape = np.array(obs).shape
            self.initialized = True
            
            # Initialize info keys based on environment type
            self.info_keys = []
            if env_base and env_base in self.env_type_keys:
                self.info_keys = self.env_type_keys[env_base].copy()
            
            # If we get actual info, add its keys too
            if info:
                for key in info.keys():
                    if key not in self.info_keys:
                        self.info_keys.append(key)
        
        # Process observation to ensure consistent shape and type
        observation = None
        if obs is not None:
            observation = np.array(obs, dtype=np.float32)
            if self.obs_shape is not None and observation.shape != self.obs_shape:
                # Reshape if needed - this should not normally happen
                try:
                    observation = np.resize(observation, self.obs_shape)
                except:
                    # If resize fails, create a zero array of right shape
                    observation = np.zeros(self.obs_shape, dtype=np.float32)
        else:
            # Create empty observation with right shape if None
            observation = np.zeros(self.obs_shape if self.obs_shape else (1,), dtype=np.float32)
            
        # Process next_observation similarly
        next_observation = None
        if next_obs is not None:
            next_observation = np.array(next_obs, dtype=np.float32)
            if self.obs_shape is not None and next_observation.shape != self.obs_shape:
                try:
                    next_observation = np.resize(next_observation, self.obs_shape)
                except:
                    next_observation = np.zeros(self.obs_shape, dtype=np.float32)
        else:
            next_observation = np.zeros(self.obs_shape if self.obs_shape else (1,), dtype=np.float32)
        
        # Process action to ensure it's a float32 numpy array
        action_array = None
        if action is not None:
            action_array = np.array(action, dtype=np.float32)
        else:
            # Create empty action if None
            action_array = np.zeros((1,), dtype=np.float32)
        
        # Process reward to ensure it's a float
        reward_value = float(reward) if reward is not None else 0.0
        
        # Process terminated and truncated to ensure they're bool
        termination_value = bool(terminated) if terminated is not None else False
        truncation_value = bool(truncated) if truncated is not None else False
        
        # Process info to ensure it has all required keys and consistent types
        info_dict = {}
        
        # Populate with default values for all known keys
        if self.info_keys:
            for key in self.info_keys:
                info_dict[key] = 0.0
        
        # Update with actual values from info (only if they're simple types)
        if info:
            for key, value in info.items():
                if isinstance(value, (int, float, bool, np.integer, np.floating, np.bool_)):
                    info_dict[key] = float(value) if isinstance(value, (float, np.floating)) else value
        
        # Ensure all environment-specific keys are present 
        if env_base and env_base in self.env_type_keys:
            for key in self.env_type_keys[env_base]:
                if key not in info_dict:
                    info_dict[key] = 0.0
        
        # Always have at least one key if info would be empty
        if not info_dict:
            info_dict["dummy"] = 0.0
            
        # Build the final step data dictionary
        step_data = {
            "observation": observation,
            "action": action_array,
            "next_observation": next_observation,
            "reward": reward_value,
            "termination": termination_value,
            "truncation": truncation_value,
            "info": info_dict,
            "env_id": env_id
        }
        
        return step_data


class TrajectoryCollector:
    """
    Collects and manages trajectories in Minari format using trained RL policies.

    This class provides functionality to:
    1. Train or load SAC policies
    2. Collect trajectories with trained or random policies
    3. Save and load datasets in Minari format for offline RL
    """
    def __init__(
        self, 
        env_name: str,
        model_path: Optional[str] = None,
        device: str = "cuda" if torch.cuda.is_available() else "cpu"
    ):
        # * Initialize environment
        self.env = gym.make(env_name)
        self.env_name = env_name
        self.device = device

        # * Load or create SAC model
        if model_path is not None and os.path.exists(model_path):
            self.model = SAC.load(model_path, env=self.env, device=self.device)
            print(f"Loaded trained SAC model from {model_path}")
        else:
            print(f"No model found at {model_path}, creating new SAC model")
            self.model = SAC("MlpPolicy", self.env, verbose=1, device=self.device)

    
    def train_model(self, total_timesteps: int = 1000000, save_path: Optional[str] = None) -> None:
        """
        Train the SAC model on the environment.
        
        Args:
            total_timesteps: Total number of timesteps to train for
            save_path: Path to save the trained model (optional)
        """
        # * Train the model for the specified number of timesteps
        self.model.learn(total_timesteps=total_timesteps)

        # * Save the trained model if a path is provided
        if save_path is not None:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            self.model.save(save_path)
            print(f"Saved trained model to {save_path}")


    def evaluate_model(self, n_eval_episodes: int = 10) -> Tuple[float, float]:
        """
        Evaluate the current model on the environment.

        Args:
            n_eval_episodes: Number of episodes to evaluate
            
        Returns:
            mean_reward: Mean reward across evaluation episodes
            std_reward: Standard deviation of rewards
        """
        # * Evaluate the model on the environment
        mean_reward, std_reward = evaluate_policy(
            self.model, 
            self.env, 
            n_eval_episodes=n_eval_episodes, 
            deterministic=True
        )
        print(f"Mean reward: {mean_reward:.2f} +/- {std_reward:.2f}")
        return mean_reward, std_reward
    

    def collect_trajectories(
        self, 
        n_episodes: int = 100, 
        deterministic: bool = True, 
        random_policy: bool = False,
        dataset_id: Optional[str] = None
    ) -> minari.DataCollector:
        """
        Collect trajectories by rolling out the current policy in the environment.
        Uses Minari's DataCollector to record the trajectories.

        Args:
            n_episodes: Number of episodes to collect
            deterministic: Whether to use deterministic actions from the policy
            random_policy: Whether to use a random policy instead of the learned one
            dataset_id: Optional ID for the dataset (if None, dataset isn't created)

        Returns:
            collector_env: DataCollector containing collected trajectories
        """
        # * Create a custom callback
        callback_instance = TypeConsistentCallback()
            
        # * Create callback class that returns our instance
        # * This workaround ensures Minari gets our initialized instance
        CallbackClass = type('DynamicCallback', (StepDataCallback,), {
            '__init__': lambda self: None,
            '__call__': lambda self, **kwargs: callback_instance(**kwargs)
        })
        
        # * Wrap the environment with Minari's DataCollector
        collector_env = DataCollector(
            self.env, 
            record_infos=True,
            step_data_callback=CallbackClass
        )
        
        # * Collect data for the specified number of episodes
        for episode in range(n_episodes):
            # * Reset the environment to get initial observation
            obs, info = collector_env.reset()
            done = False
            episode_length = 0

            # * Run one episode
            while not done:
                # * Select action based on the specified policy type
                if random_policy:
                    action = self.env.action_space.sample()
                else:
                    action, _ = self.model.predict(obs, deterministic=deterministic)
                
                # * Step the environment using the selected action
                next_obs, reward, terminated, truncated, info = collector_env.step(action)
                
                # * Update current observation and check if episode is done
                obs = next_obs
                done = terminated or truncated
                episode_length += 1

            print(f"Episode {episode+1}/{n_episodes} completed with length {episode_length}")

        # * Create a Minari dataset if dataset_id is provided
        if dataset_id is not None:
            dataset = collector_env.create_dataset(
                dataset_id=dataset_id,
                algorithm_name="SAC" if not random_policy else "Random",
                code_permalink="https://github.com/yourusername/your-repository",
                author="Your Name",
            )
            return dataset
        else:
            # * Return the collector environment for further processing
            return collector_env
    

    def save_dataset_minari_format(
        self, 
        collector_env: DataCollector, 
        dataset_id: str
    ) -> minari.MinariDataset:
        """
        Save the collected trajectories in Minari format.
        
        Args:
            collector_env: Minari DataCollector environment with collected data
            dataset_id: ID for the dataset
            
        Returns:
            dataset: The created Minari dataset
        """
        # * Create and save the dataset using Minari
        dataset = collector_env.create_dataset(
            dataset_id=dataset_id,
            algorithm_name="SAC",
            code_permalink="https://github.com/yourusername/your-repository",
            author="Your Name"
        )
        
        print(f"Dataset saved with ID: {dataset_id}")
        return dataset


    def load_dataset_from_minari(self, dataset_id: str) -> minari.MinariDataset:
        """
        Load trajectories from a Minari dataset.
        
        Args:
            dataset_id: ID of the Minari dataset to load
                
        Returns:
            dataset: Loaded Minari dataset
        """
        # * Check if dataset exists locally, otherwise try to download
        try:
            dataset = minari.load_dataset(dataset_id)
        except ValueError:
            print(f"Dataset not found locally. Trying to download from remote...")
            dataset = minari.load_dataset(dataset_id, download=True)
        
        print(f"Dataset loaded: {dataset_id}")
        return dataset
    

    def get_qlearning_dataset(self, dataset: minari.MinariDataset) -> Dict[str, np.ndarray]:
        """
        Convert a Minari dataset to Q-learning format (with next_observations).
        Replacement for the d4rl.qlearning_dataset function.
        
        Args:
            dataset: Minari dataset to convert
                
        Returns:
            qlearning_dataset: Dictionary containing the dataset in Q-learning format
        """
        # * Initialize lists to store the combined data
        observations = []
        next_observations = []
        actions = []
        rewards = []
        terminals = []
        
        # * Iterate through each episode in the dataset
        for episode_data in dataset.iterate_episodes():
            ep_obs = episode_data.observations
            ep_actions = episode_data.actions
            ep_rewards = episode_data.rewards
            ep_terminals = episode_data.terminations
            
            # * Add data from this episode
            observations.extend(ep_obs[:-1])  # Exclude the last observation
            next_observations.extend(ep_obs[1:])  # Start from the second observation
            actions.extend(ep_actions[:-1])  # Exclude the last action
            rewards.extend(ep_rewards[:-1])  # Exclude the last reward
            terminals.extend(ep_terminals[:-1])  # Exclude the last termination flag
            
        # * Convert lists to numpy arrays
        qlearning_dataset = {
            'observations': np.array(observations),
            'next_observations': np.array(next_observations),
            'actions': np.array(actions),
            'rewards': np.array(rewards),
            'terminals': np.array(terminals)
        }
        
        print(f"Created Q-learning dataset with {len(observations)} transitions")
        return qlearning_dataset
