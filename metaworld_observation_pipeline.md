# MetaWorld Observation Pipeline for SAC

This document provides a detailed explanation of how observations are generated and processed in the MetaWorld environment for use with the SAC (Soft Actor-Critic) algorithm.

## Overview

The observation pipeline consists of three main stages:
1. **Raw Observation Generation** - MetaWorld environments generate base observations
2. **Multi-task Wrapping** - Observations are wrapped with task information  
3. **SAC Agent Processing** - SAC agent processes observations for decision making

## 1. Raw Observation Generation (MetaWorld Base Environment)

### Location: `Metaworld/metaworld/envs/mujoco/sawyer_xyz/sawyer_xyz_env.py`

MetaWorld environments generate observations through two key methods:

### `_get_curr_obs_combined_no_goal()` - Base Observation (18 elements)

This method creates the core observation containing:

```python
def _get_curr_obs_combined_no_goal(self) -> npt.NDArray[np.float64]:
    # 1. End effector position (3D coordinates)
    pos_hand = self.get_endeff_pos()
    
    # 2. Gripper state (normalized distance between fingers)
    finger_right, finger_left = self.data.body("rightclaw"), self.data.body("leftclaw")
    gripper_distance_apart = np.linalg.norm(finger_right.xpos - finger_left.xpos)
    gripper_distance_apart = np.clip(gripper_distance_apart / 0.1, 0.0, 1.0)
    
    # 3. Object information (positions + quaternions, padded to 14 elements)
    obs_obj_padded = np.zeros(self._obs_obj_max_len)  # max 14 elements
    obj_pos = self._get_pos_objects()  # 3D positions
    obj_quat = self._get_quat_objects()  # 4D quaternions
    
    # Combine: [pos_hand(3) + gripper(1) + objects(14)] = 18 elements
    return np.hstack((pos_hand, gripper_distance_apart, obs_obj_padded))
```

**Observation Structure (18 elements):**
- Elements 0-2: End effector XYZ position
- Element 3: Gripper opening distance (normalized 0-1)
- Elements 4-17: Object positions and orientations (padded)

### `_get_obs()` - Full Observation with Frame Stacking (39 elements)

This method adds temporal information and goal:

```python
def _get_obs(self) -> npt.NDArray[np.float64]:
    # 1. Get goal position (3D)
    pos_goal = self._get_pos_goal()
    if self._partially_observable:
        pos_goal = np.zeros_like(pos_goal)  # Hide goal in partially observable tasks
    
    # 2. Current observation
    curr_obs = self._get_curr_obs_combined_no_goal()  # 18 elements
    
    # 3. Frame stacking: current + previous + goal
    obs = np.hstack((curr_obs, self._prev_obs, pos_goal))  # 18 + 18 + 3 = 39
    self._prev_obs = curr_obs
    
    return obs
```

**Full Observation Structure (39 elements):**
- Elements 0-17: Current observation (end effector, gripper, objects)
- Elements 18-35: Previous observation (frame stacking for temporal information)
- Elements 36-38: Goal position (XYZ coordinates)

## 2. Multi-task Wrapping (Vector Environment)

### Location: `mtrl/env/vec_env.py`

The `MetaWorldVecEnv` class wraps raw observations with task information:

```python
class MetaWorldVecEnv(AsyncVectorEnv):
    def __init__(self, env_metadata, env_fns, ...):
        super().__init__(...)
        self.task_obs = torch.arange(self.num_envs)  # Task indices as tensor
    
    def create_multitask_obs(self, env_obs):
        return {
            "env_obs": torch.tensor(env_obs),      # Raw environment observations  
            "task_obs": self.task_obs              # Task indices [0, 1, 2, ...]
        }
    
    def reset(self):
        env_obs = super().reset()
        return self.create_multitask_obs(env_obs=env_obs[0])
    
    def step(self, actions):
        env_obs, reward, done, truncated, info = super().step(actions)
        return self.create_multitask_obs(env_obs=env_obs), reward, done, info
```

**Multi-task Observation Structure:**
```python
{
    "env_obs": torch.Tensor,  # Shape: [num_envs, 39] - Raw MetaWorld observations
    "task_obs": torch.Tensor  # Shape: [num_envs] - Task indices [0, 1, 2, ...]
}
```

## 3. SAC Agent Processing

### Location: `mtrl/agent/sac.py`

The SAC agent processes multi-task observations in its `act()` method:

```python
def act(self, multitask_obs: ObsType, modes: List[str], sample: bool = True) -> np.ndarray:
    # Extract components from multi-task observation
    env_obs = multitask_obs["env_obs"]      # Raw environment observations
    env_index = multitask_obs["task_obs"]   # Task indices
    
    # Process each environment
    for i, mode in enumerate(modes):
        if mode == "train":
            # Create task info for this environment/task
            task_info = TaskInfo(task_obs=env_index[i].item())
            
            # Prepare observation tensor
            obs = env_obs.float().to(self.device)
            if len(obs.shape) == 1 or len(obs.shape) == 3:
                obs = obs.unsqueeze(0)  # Add batch dimension
            
            # Create MTObs datastructure
            mtobs = MTObs(
                env_obs=obs,           # Raw observation (39 elements)
                task_obs=env_index,    # Task information
                task_info=task_info    # Additional task metadata
            )
            
            # Get action from actor network
            mu, pi, _, _ = self.actor(mtobs=mtobs)
            action = pi if sample else mu
```

### MTObs Datastructure

The `MTObs` class (in `mtrl/agent/ds/mt_obs.py`) combines all observation information:

```python
@dataclass
class MTObs:
    env_obs: TensorType      # Raw environment observation (39 elements)
    task_obs: TensorType     # Task indices 
    task_info: TaskInfo      # Additional task metadata
```

## Environment Construction Pipeline

### Location: `mtrl/env/builder.py`

Environments are built through the `build_metaworld_vec_env()` function:

```python
def build_metaworld_vec_env(config, benchmark, mode, env_id_to_task_map):
    # Extract benchmark info
    benchmark_name = config.env.benchmark._target_.replace("metaworld.", "")
    num_tasks = int(benchmark_name.replace("MT", ""))  # e.g., MT10 → 10 tasks
    
    # Create environment functions
    funcs_to_make_envs, env_id_to_task_map = get_list_of_func_to_make_metaworld_envs(
        benchmark=benchmark,
        benchmark_name=benchmark_name,
        env_id_to_task_map=env_id_to_task_map,
        should_perform_reward_normalization=True
    )
    
    # Create vectorized environment
    env = MetaWorldVecEnv(
        env_metadata={
            "ids": list(range(num_tasks)),
            "mode": [mode for _ in range(num_tasks)]
        },
        env_fns=funcs_to_make_envs
    )
    
    return env, env_id_to_task_map
```

## Summary

The complete observation flow is:

1. **MetaWorld Environment** generates 39-element observations:
   - 18 current elements (end effector + gripper + objects)
   - 18 previous elements (frame stacking)
   - 3 goal elements

2. **MetaWorldVecEnv** wraps with task information:
   - `env_obs`: Raw observations as tensor
   - `task_obs`: Task indices as tensor

3. **SAC Agent** processes observations:
   - Extracts environment and task observations
   - Creates `MTObs` datastructure
   - Passes to actor network for action selection

This pipeline enables the SAC agent to handle multi-task learning across different MetaWorld manipulation tasks while maintaining task-specific information for context.