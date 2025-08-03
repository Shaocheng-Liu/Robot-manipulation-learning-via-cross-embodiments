# 🤖 Comprehensive Training Guide for Robot Manipulation Learning

This guide provides a complete explanation of the training pipeline for multi-robot manipulation learning via cross-embodiments.

## 🔍 Robot Observation Analysis: Sawyer vs UR10e

### Observation Dimension Consistency

**Answer to Question**: Yes, both Sawyer and UR10e robots output the **same dimensional state** (39 elements).

**Key Finding**: The observation structure is defined in `Metaworld/metaworld/envs/mujoco/sawyer_xyz/sawyer_xyz_env.py` and is **robot-agnostic**. Different robots use the same observation pipeline:

```python
# Observation Structure (39 elements total):
def _get_obs(self) -> np.ndarray:
    # Current observation (18 elements)
    curr_obs = self._get_curr_obs_combined_no_goal()  # 18 elements
    # Frame stacking (18 elements) 
    prev_obs = self._prev_obs                         # 18 elements
    # Goal information (3 elements)
    pos_goal = self._get_pos_goal()                   # 3 elements
    
    # Total: 18 + 18 + 3 = 39 elements
    return np.hstack((curr_obs, prev_obs, pos_goal))

# Current observation breakdown (18 elements):
def _get_curr_obs_combined_no_goal(self) -> np.ndarray:
    pos_hand = self.get_endeff_pos()              # End effector position (3)
    gripper_distance_apart = ...                  # Gripper state (1)
    obs_obj_padded = ...                          # Object information (14)
    
    # Total: 3 + 1 + 14 = 18 elements
    return np.hstack((pos_hand, gripper_distance_apart, obs_obj_padded))
```

**Robot Assets**: Different robots use different XML files (`xyz_base_ur5e.xml`, `xyz_base_ur10e.xml`, `sawyer_*.xml`) but all follow the same observation interface.

---

## 🔄 Complete Training Pipeline Explanation

### Overview
The training consists of **4 main phases**:
1. **Expert Training** (`train_task`) - Train individual robot experts per task
2. **Data Generation** (`online_distill`) - Generate demonstrations from experts  
3. **Data Preparation** (`split_buffer`) - Split data for transformer training
4. **Collective Learning** - Train shared policies across robots

---

## 📋 Detailed Mode Explanations

### 1. `train_task` - Expert Training Phase

**Purpose**: Train individual SAC agents (experts) for each task-robot combination.

**Code**:
```bash
train_task reach-v2 100000 worker.builder.actor_update_freq=1
```

**What it does**:
- Trains a SAC agent on a specific task (`reach-v2`) for 100,000 steps
- Uses mode `experiment.mode=train_worker` 
- Saves model to: `logs/experiment_test/model_dir/model_${task_name}_seed_1/`

**Input**: Environment + task configuration
**Output**: 
- Expert model: `model_${task_name}_seed_1/`
- Replay buffer: `buffer/buffer_${task_name}_seed_1/`
- **Special buffers**: `buffer_distill_${task_name}_seed_1/` and `buffer_distill_tmp_${task_name}_seed_1/`

### 2. `online_distill` - Data Generation Phase

**Purpose**: Use trained experts to generate demonstration data for collective learning.

**Code**:
```bash
online_distill reach-v2
```

**What it does**:
- Loads expert model for the task
- Uses mode `experiment.mode=online_distill_collective_transformer`
- Runs expert policy to collect high-quality demonstrations
- Generates data for training the collective transformer network

**Input**: Expert model from phase 1
**Output**: `online_buffer_${task_name}/` - demonstrations for collective training

### 3. `split_buffer` - Data Preparation Phase

**Purpose**: Split collected data into training/validation sets for transformer training.

**Code**:
```bash
split_buffer reach-v2
```

**What it does**:
- Processes `buffer_distill_${task_name}_seed_1/`
- Splits into train/validation sets
- Prepares data for trajectory transformer training

**Input**: `buffer_distill_${task_name}_seed_1/`
**Output**: 
- `Transformer_RNN/dataset/train/buffer_distill_${task_name}_seed_1/`
- `Transformer_RNN/dataset/validation/buffer_distill_${task_name}_seed_1/`

### 4. `split_online_buffer` - Collective Data Preparation

**Purpose**: Prepare demonstration data for collective network training.

**Code**:
```bash
split_online_buffer reach-v2
```

**Input**: `online_buffer_${task_name}/`
**Output**:
- `logs/experiment_test/buffer/collective_buffer/train/online_buffer_${task_name}_seed_1/`
- `logs/experiment_test/buffer/collective_buffer/validation/online_buffer_${task_name}_seed_1/`

---

## 🗂️ Buffer Files Explanation

### `buffer_distill` and `buffer_distill_tmp`

**Purpose**: These are **essential** for the collective learning pipeline.

- **`buffer_distill`**: Contains processed demonstration data from expert policies
- **`buffer_distill_tmp`**: Temporary storage during data collection

**Generation**: Created during `train_task` phase when experts reach certain performance thresholds.

**Requirement**: **YES**, these buffers are required for `online_distill` and subsequent collective training phases.

**Why needed**: The collective transformer needs high-quality demonstrations from multiple robot experts to learn shared representations.

---

## 🔧 Complete Training Workflow

### Phase 1: Train Individual Experts
```bash
# Train experts on all tasks
train_task reach-v2 100000 worker.builder.actor_update_freq=1
train_task push-v2 900000
train_task pick-place-v2 2400000
# ... more tasks
```

### Phase 2: Generate Demonstrations
```bash
# Generate demonstrations from experts
online_distill reach-v2
online_distill push-v2
online_distill pick-place-v2
# ... more tasks
```

### Phase 3: Prepare Data for Transformers
```bash
# Split data for trajectory transformer
split_buffer reach-v2
split_buffer push-v2
# ... more tasks

# Split data for collective network
split_online_buffer reach-v2
split_online_buffer push-v2
# ... more tasks
```

### Phase 4: Train Collective Models
```bash
# Train trajectory transformer
python3 Transformer_RNN/dataset_tf.py
python3 Transformer_RNN/RepresentationTransformerWithCLS.py

# Train collective network
python3 -u main.py setup=metaworld env=metaworld-mt1 \
    experiment.mode=distill_collective_transformer
```

### Phase 5: Evaluation
```bash
# Evaluate individual experts
evaluate_task reach-v2
evaluate_task push-v2

# Evaluate collective network
evaluate_col_agent reach-v2
evaluate_col_agent push-v2
```

---

## 🛠️ File Dependencies Summary

| Phase | Input Files | Output Files | Required For |
|-------|------------|--------------|--------------|
| `train_task` | Environment config | `model_${task}_seed_1/`<br>`buffer_distill_${task}_seed_1/` | All subsequent phases |
| `online_distill` | Expert models | `online_buffer_${task}/` | Collective training |
| `split_buffer` | `buffer_distill_*` | Train/val datasets | Trajectory transformer |
| `split_online_buffer` | `online_buffer_*` | Collective train/val | Collective network |
| Collective training | All above | `model_col/` | Final evaluation |

---

## 🤖 Multi-Robot Model Naming Solution

### Current Issue
Currently, models are saved as `model_${task_name}_seed_1` regardless of robot type, causing conflicts when training the same task on different robots.

### Solution: Robot-Aware Model Naming

**Location to Modify**: `mtrl/experiment/collective_learning.py` in the model saving logic.

**Current Code Pattern**:
```python
# In model saving logic (approximate location)
model_path = f"model_{task_name}_seed_{seed}"
```

**Modified Code Pattern**:
```python
# Add robot identification to model naming
robot_type = self.get_robot_type()  # e.g., 'sawyer', 'ur5e', 'ur10e'
model_path = f"model_{robot_type}_{task_name}_seed_{seed}"
```

### Implementation Steps

1. **Add Robot Detection Method**:
```python
def get_robot_type(self) -> str:
    """Detect robot type from environment configuration."""
    # Check asset files or environment config to determine robot
    # Could be based on XML files used or config parameters
    return "sawyer"  # Default, but implement detection logic
```

2. **Modify Model Directory Creation**:
```python
# In save model methods
model_dir = f"{base_path}/model_{robot_type}_{task_name}_seed_{seed}"
```

3. **Update Buffer Naming**:
```python
# Update buffer names to include robot type
buffer_path = f"buffer_{robot_type}_{task_name}_seed_{seed}"
buffer_distill_path = f"buffer_distill_{robot_type}_{task_name}_seed_{seed}"
```

### Expected Result
- **Before**: `model_reach-v2_seed_1` (conflicts between robots)
- **After**: `model_sawyer_reach-v2_seed_1`, `model_ur10e_reach-v2_seed_1` (robot-specific)

This enables training multiple robot experts per task and using them together in collective learning.

---

## 📊 Training Monitoring

### Success Metrics
- Individual expert success rate > 90%
- Collective network maintains performance across robots
- Successful knowledge transfer between robot embodiments

### Log Locations
- Model checkpoints: `logs/experiment_test/model_dir/`
- Evaluation results: `logs/results/`
- Training logs: Console output during training

---

## 🚀 Quick Start Commands

```bash
# Complete training pipeline for a single task
task_name="reach-v2"
steps=100000

# 1. Train expert
train_task $task_name $steps

# 2. Generate demonstrations  
online_distill $task_name

# 3. Prepare data
split_buffer $task_name
split_online_buffer $task_name

# 4. Evaluate expert
evaluate_task $task_name
```

This guide provides the complete understanding needed to train and evaluate multi-robot manipulation policies using the cross-embodiment learning framework.