"""
    This script generates D4RL format datasets according to configurations.
"""


"""
    Standard Libraries
"""
import argparse
import os
import sys
from typing import Optional

# Set MuJoCo environment variable programmatically
mujoco_path = os.path.expanduser('~/.mujoco/mujoco210/bin')
if os.path.exists(mujoco_path):
    if 'LD_LIBRARY_PATH' in os.environ:
        os.environ['LD_LIBRARY_PATH'] = f"{os.environ['LD_LIBRARY_PATH']}:{mujoco_path}"
    else:
        os.environ['LD_LIBRARY_PATH'] = mujoco_path
    print(f"Set LD_LIBRARY_PATH to include {mujoco_path}")

"""
    3rd Party Libraries
"""
import minari
from minari import DataCollector
import numpy as np
import torch

"""
    Files
"""
from data_collection.utils.config import load_config
from data_collection.generate_d4rl_datasets import TrajectoryCollector


def generate_datasets(config_path: str, env_filter: Optional[str] = None) -> None:
    """
        Generate datasets based on configuration file.
        
        Args:
            config_path: Path to the configuration file
            env_filter: If specified, only process this environment
    """
    # * Get the directory of the current script to save all outputs there
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # * Resolve the config file path
    # * First, check if it's an absolute path
    if not os.path.isabs(config_path):
        # * Check if the file exists in the current directory
        if not os.path.exists(config_path):
            # * Try to find it relative to the script directory
            script_relative_path = os.path.join(script_dir, config_path)
            
            if os.path.exists(script_relative_path):
                config_path = script_relative_path
            else:
                # * Also try looking in the parent directory
                parent_dir = os.path.dirname(script_dir)
                parent_relative_path = os.path.join(parent_dir, config_path)
                
                if os.path.exists(parent_relative_path):
                    config_path = parent_relative_path
    
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    print(f"Using config file: {config_path}")
    
    # * Load configuration
    config = load_config(config_path)

    # * Get global settings
    settings = config.get('settings', {})
    models_dir_name = settings.get('models_dir', 'models')
    datasets_dir_name = settings.get('datasets_dir', 'datasets')
    eval_episodes = settings.get('eval_episodes', 10)
    device = settings.get('device', 'cuda' if torch.cuda.is_available() else 'cpu')
    
    # * Create namespace for datasets (optional)
    namespace = settings.get('namespace', None)
    
    # * Prefix for dataset IDs
    dataset_prefix = f"{namespace}/" if namespace else ""

    # * Make directories relative to the script location, not the config file or working directory
    models_dir = os.path.join(script_dir, models_dir_name)
    datasets_dir = os.path.join(script_dir, datasets_dir_name)
    
    print(f"Models will be saved to: {models_dir}")
    print(f"Datasets will be saved to: {datasets_dir}")

    # * Create output directories
    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(datasets_dir, exist_ok=True)

    # * Dictionary to store collector environments for mixed dataset creation
    env_collectors = {}

    # * Process each environment
    for env_config in config.get('environments', []):
        env_name = env_config.get('name')

        # * Skip if env_filter is specified and doesn't match
        if env_filter and env_filter != env_name:
            continue

        print(f"\n{'='*80}\nProcessing environment: {env_name}\n{'='*80}")

        # * Store collectors for this environment
        env_collectors[env_name] = {}

        # * Create collector
        collector = TrajectoryCollector(env_name, device=device)

        # * Train model for this environment
        train_steps = env_config.get('train_steps', 1000000)
        model_path = os.path.join(models_dir, f"{env_name}_sac.zip")

        print(f"Training SAC model for {train_steps} steps...")
        collector.train_model(total_timesteps=train_steps, save_path=model_path)

        # * Evaluate the trained model
        print(f"Evaluating trained model...")
        collector.evaluate_model(n_eval_episodes=eval_episodes)

        # * Process each dataset configuration
        for dataset_config in env_config.get('datasets', []):
            dataset_type = dataset_config.get('type')

            # * Skip mixed datasets for now (handle later)
            if dataset_type == 'mixed':
                continue

            print(f"\nGenerating {dataset_type} dataset for {env_name}...")

            # * Special handling for medium datasets (partially trained model)
            if dataset_type == 'medium':
                checkpoint_steps = dataset_config.get('checkpoint', train_steps // 2)
                medium_model_path = os.path.join(models_dir, f"{env_name}_sac_medium.zip")

                # * Create a new collector with fresh model
                medium_collector = TrajectoryCollector(env_name, device=device)

                # * Train to medium performance
                print(f"Training medium performance model for {checkpoint_steps} steps...")
                medium_collector.train_model(total_timesteps=checkpoint_steps, save_path=medium_model_path)
                
                # * Collect trajectories with medium model
                episodes = dataset_config.get('episodes', 100)
                deterministic = dataset_config.get('deterministic', False)
                random_policy = dataset_config.get('random_policy', False)
                
                # * Create dataset ID
                dataset_id = f"{dataset_prefix}{env_name}/{dataset_type}-v0"

                # * Collect trajectories and create dataset
                collector_env = medium_collector.collect_trajectories(
                    n_episodes=episodes,
                    deterministic=deterministic,
                    random_policy=random_policy
                )
                
                # * Save dataset
                dataset = medium_collector.save_dataset_minari_format(
                    collector_env,
                    dataset_id=dataset_id
                )
                
                # * Store collector for potential mixed dataset creation
                env_collectors[env_name][dataset_type] = collector_env

            else:
                # * Standard dataset collection (expert or random)
                episodes = dataset_config.get('episodes', 100)
                deterministic = dataset_config.get('deterministic', dataset_type == 'expert')
                random_policy = dataset_config.get('random_policy', dataset_type == 'random')
                
                # * Create dataset ID
                dataset_id = f"{dataset_prefix}{env_name}/{dataset_type}-v0"

                # * Collect trajectories
                collector_env = collector.collect_trajectories(
                    n_episodes=episodes,
                    deterministic=deterministic,
                    random_policy=random_policy
                )
                
                # * Save dataset
                dataset = collector.save_dataset_minari_format(
                    collector_env,
                    dataset_id=dataset_id
                )
                
                # * Store collector for potential mixed dataset creation
                env_collectors[env_name][dataset_type] = collector_env

        # * Now create mixed datasets if specified
        for dataset_config in env_config.get('datasets', []):
            if dataset_config.get('type') == 'mixed':
                print(f"\nGenerating mixed dataset for {env_name}...")

                # * Get source datasets and mixing ratio
                source_datasets = dataset_config.get('source_datasets', ['expert', 'random'])
                expert_ratio = dataset_config.get('expert_ratio', 0.5)
                
                # * Create dataset ID
                dataset_id = f"{dataset_prefix}{env_name}/mixed-v0"
                
                # * Ensure source datasets exist
                if not all(ds in env_collectors[env_name] for ds in source_datasets):
                    print(f"Error: Not all source datasets {source_datasets} available for {env_name}")
                    continue
                
                # * For mixed datasets, we need to load the individual datasets and sample episodes
                # * from each according to the mixing ratio
                try:
                    # * Load the source datasets
                    source1_id = f"{dataset_prefix}{env_name}/{source_datasets[0]}-v0"
                    source2_id = f"{dataset_prefix}{env_name}/{source_datasets[1]}-v0"
                    
                    dataset1 = minari.load_dataset(source1_id)
                    dataset2 = minari.load_dataset(source2_id)
                    
                    # * Get total episodes counts
                    n_episodes1 = dataset1.total_episodes
                    n_episodes2 = dataset2.total_episodes
                    
                    # * Calculate number of episodes to take from each source
                    # * based on the mixing ratio
                    episodes_to_take_1 = int(n_episodes1 * expert_ratio)
                    episodes_to_take_2 = int(n_episodes2 * (1 - expert_ratio))
                    
                    print(f"Creating mixed dataset with {episodes_to_take_1} episodes from {source_datasets[0]} "
                          f"and {episodes_to_take_2} episodes from {source_datasets[1]}")
                          
                    # * Note: This is a simplified approach. In a complete implementation,
                    # * you would sample episodes from both datasets and create a new merged dataset.
                    # * This would require access to Minari's internal API for creating datasets
                    # * from episode buffers.
                    
                    print(f"Note: Full implementation of mixed dataset creation would require "
                          f"sampling episodes from both source datasets and creating a new dataset. "
                          f"This functionality is not fully implemented in this script.")
                
                except Exception as e:
                    print(f"Error creating mixed dataset: {e}")


def main():
    parser = argparse.ArgumentParser(description='Generate D4RL format datasets')
    parser.add_argument('--config', type=str, default='dataset_config.yaml', 
                        help='Path to configuration file')
    parser.add_argument('--env', type=str, default=None,
                        help='Only process this environment (optional)')
    args = parser.parse_args()

    generate_datasets(args.config, args.env)


if __name__ == '__main__':
    main()

