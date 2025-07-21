#!/usr/bin/env python3
"""
Example demonstration of how MetaWorld observations flow through the SAC pipeline.

This script shows the key steps in observation processing from MetaWorld environment
to SAC agent decision making.
"""

import numpy as np
import torch
from typing import Dict, List, Any

# Simulated observation flow demonstration

def simulate_metaworld_observation():
    """
    Simulate MetaWorld environment observation generation.
    
    This represents what happens in:
    Metaworld/metaworld/envs/mujoco/sawyer_xyz/sawyer_xyz_env.py
    """
    
    print("=== Step 1: MetaWorld Base Environment Observation Generation ===")
    
    # 1. Get current state (18 elements)
    end_effector_pos = np.array([0.15, 0.6, 0.2])  # 3D position
    gripper_distance = np.array([0.8])  # normalized 0-1
    object_info = np.random.randn(14)  # positions + quaternions (padded)
    
    current_obs = np.hstack([end_effector_pos, gripper_distance, object_info])
    print(f"Current observation (18 elements): shape={current_obs.shape}")
    print(f"  - End effector: {end_effector_pos}")
    print(f"  - Gripper: {gripper_distance[0]:.3f}")
    print(f"  - Objects: {object_info[:4]} ... (truncated)")
    
    # 2. Add frame stacking + goal (39 elements total)
    previous_obs = np.random.randn(18)  # Previous timestep observation
    goal_pos = np.array([0.2, 0.65, 0.15])  # 3D goal position
    
    full_obs = np.hstack([current_obs, previous_obs, goal_pos])
    print(f"\nFull observation with frame stacking (39 elements): shape={full_obs.shape}")
    print(f"  - Current: {current_obs.shape[0]} elements")
    print(f"  - Previous: {previous_obs.shape[0]} elements") 
    print(f"  - Goal: {goal_pos.shape[0]} elements")
    
    return full_obs

def simulate_multitask_wrapping(env_observations: np.ndarray, num_envs: int = 3):
    """
    Simulate multi-task observation wrapping.
    
    This represents what happens in:
    mtrl/env/vec_env.py - MetaWorldVecEnv.create_multitask_obs()
    """
    
    print(f"\n=== Step 2: Multi-task Observation Wrapping ===")
    
    # Simulate multiple parallel environments
    batch_env_obs = np.tile(env_observations, (num_envs, 1))  # [num_envs, 39]
    task_indices = np.arange(num_envs)  # [0, 1, 2] for MT3 example
    
    multitask_obs = {
        "env_obs": torch.tensor(batch_env_obs, dtype=torch.float32),
        "task_obs": torch.tensor(task_indices, dtype=torch.long)
    }
    
    print(f"Multi-task observation dict:")
    print(f"  - env_obs: shape={multitask_obs['env_obs'].shape} dtype={multitask_obs['env_obs'].dtype}")
    print(f"  - task_obs: shape={multitask_obs['task_obs'].shape} dtype={multitask_obs['task_obs'].dtype}")
    print(f"  - Task indices: {multitask_obs['task_obs'].numpy()}")
    
    return multitask_obs

def simulate_sac_processing(multitask_obs: Dict[str, torch.Tensor]):
    """
    Simulate SAC agent observation processing.
    
    This represents what happens in:
    mtrl/agent/sac.py - Agent.act()
    """
    
    print(f"\n=== Step 3: SAC Agent Observation Processing ===")
    
    # Extract observations (as done in SAC agent)
    env_obs = multitask_obs["env_obs"]      # Raw environment observations
    env_index = multitask_obs["task_obs"]   # Task indices
    
    print(f"Extracted observations:")
    print(f"  - Environment obs: {env_obs.shape}")
    print(f"  - Task indices: {env_index.shape}")
    
    # Process each environment (simplified version)
    num_envs = env_obs.shape[0]
    actions = []
    
    for i in range(num_envs):
        # Get observation for this environment
        obs_i = env_obs[i]  # Shape: [39]
        task_i = env_index[i].item()  # Scalar task index
        
        print(f"\n  Environment {i} (Task {task_i}):")
        print(f"    - Obs shape: {obs_i.shape}")
        print(f"    - End effector pos: {obs_i[:3].numpy()}")
        print(f"    - Gripper state: {obs_i[3].item():.3f}")
        print(f"    - Goal position: {obs_i[-3:].numpy()}")
        
        # Simulate MTObs creation and actor forward pass
        # In real code: mtobs = MTObs(env_obs=obs_i, task_obs=task_i, task_info=task_info)
        # In real code: mu, pi, _, _ = self.actor(mtobs=mtobs)
        
        # Simulate action generation
        simulated_action = np.random.randn(4)  # 4D action for Sawyer
        actions.append(simulated_action)
        print(f"    - Generated action: {simulated_action}")
    
    return np.array(actions)

def main():
    """Demonstrate the complete observation pipeline."""
    
    print("MetaWorld Observation Pipeline for SAC - Example Flow")
    print("=" * 60)
    
    # Step 1: MetaWorld environment generates raw observations
    raw_obs = simulate_metaworld_observation()
    
    # Step 2: Multi-task wrapper adds task information
    multitask_obs = simulate_multitask_wrapping(raw_obs, num_envs=3)
    
    # Step 3: SAC agent processes observations for action selection
    actions = simulate_sac_processing(multitask_obs)
    
    print(f"\n=== Final Result ===")
    print(f"Generated {len(actions)} actions for {len(actions)} environments")
    print(f"Action shape per environment: {actions[0].shape}")
    
    print(f"\n=== Summary ===")
    print("1. MetaWorld env: 39-element obs (current + previous + goal)")
    print("2. Multi-task wrapper: Adds task indices for parallel envs")  
    print("3. SAC agent: Processes obs + task info → actions")

if __name__ == "__main__":
    main()