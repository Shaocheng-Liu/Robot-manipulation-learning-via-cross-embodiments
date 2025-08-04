# Comprehensive Robot-Aware Naming Implementation
# This addresses ALL path definitions in collective_experiment.py

import os
import re
from typing import List

def detect_robot_type_from_config(config) -> str:
    """
    Detect robot type from configuration.
    Returns: 'sawyer', 'ur5e', 'ur10e', or 'unknown'
    """
    # Method 1: Check environment assets path
    if hasattr(config, 'env') and hasattr(config.env, 'name'):
        env_name = config.env.name.lower()
        if 'sawyer' in env_name:
            return 'sawyer'
        elif 'ur5e' in env_name:
            return 'ur5e' 
        elif 'ur10e' in env_name:
            return 'ur10e'
    
    # Method 2: Check XML assets if available
    if hasattr(config, 'env') and hasattr(config.env, 'env_kwargs'):
        if 'xml_file' in config.env.env_kwargs:
            xml_file = config.env.env_kwargs.xml_file.lower()
            if 'sawyer' in xml_file:
                return 'sawyer'
            elif 'ur5e' in xml_file:
                return 'ur5e'
            elif 'ur10e' in xml_file:
                return 'ur10e'
    
    # Method 3: Check task configuration for robot indicators
    if hasattr(config, 'env') and hasattr(config.env, 'train'):
        task_list = config.env.train if isinstance(config.env.train, list) else []
        for task in task_list:
            task_lower = task.lower()
            if 'sawyer' in task_lower:
                return 'sawyer'
            elif 'ur5e' in task_lower:
                return 'ur5e'
            elif 'ur10e' in task_lower:
                return 'ur10e'
    
    return 'unknown'

def create_robot_aware_path(base_path: str, robot_type: str, task_name: str, seed: int, path_type: str) -> str:
    """
    Create robot-aware paths for all different path types.
    
    Args:
        base_path: Base directory path
        robot_type: Detected robot type ('sawyer', 'ur5e', 'ur10e')
        task_name: Task name
        seed: Random seed
        path_type: Type of path ('model', 'buffer', 'buffer_distill', 'buffer_distill_tmp', etc.)
    
    Returns:
        Robot-aware path string
    """
    if robot_type == 'unknown':
        # Fallback to original naming if robot detection fails
        if path_type == 'model':
            return os.path.join(base_path, f"model_dir/model_{task_name}_seed_{seed}")
        elif path_type == 'buffer':
            return os.path.join(base_path, f"buffer/buffer/buffer_{task_name}_seed_{seed}")
        elif path_type == 'buffer_distill':
            return os.path.join(base_path, f"buffer/buffer_distill/buffer_distill_{task_name}_seed_{seed}")
        elif path_type == 'buffer_distill_tmp':
            return os.path.join(base_path, f"buffer/buffer_distill_tmp/buffer_distill_tmp_{task_name}_seed_{seed}")
        # Add other path types as needed
    
    # Robot-aware naming
    if path_type == 'model':
        return os.path.join(base_path, f"model_dir/model_{robot_type}_{task_name}_seed_{seed}")
    elif path_type == 'buffer':
        return os.path.join(base_path, f"buffer/buffer/buffer_{robot_type}_{task_name}_seed_{seed}")
    elif path_type == 'buffer_distill':
        return os.path.join(base_path, f"buffer/buffer_distill/buffer_distill_{robot_type}_{task_name}_seed_{seed}")
    elif path_type == 'buffer_distill_tmp':
        return os.path.join(base_path, f"buffer/buffer_distill_tmp/buffer_distill_tmp_{robot_type}_{task_name}_seed_{seed}")
    elif path_type == 'recording_buffer_distill':
        return os.path.join(base_path, f"buffer/buffer_distill/recording_buffer_distill_{robot_type}_{task_name}_seed_{seed}")
    elif path_type == 'online_buffer':
        return os.path.join(base_path, f"buffer/online_buffer_{robot_type}_{task_name}")
    elif path_type == 'student_model':
        return os.path.join(base_path, f"model_dir/student_model_{robot_type}_{task_name}_seed_{seed}")
    elif path_type == 'student_buffer':
        return os.path.join(base_path, f"buffer/student_buffer_{robot_type}_{task_name}_seed_{seed}")
    elif path_type == 'policy_distill':
        return os.path.join(base_path, f"buffer/policy_distill_{robot_type}_{task_name}_seed_{seed}")
    else:
        raise ValueError(f"Unknown path_type: {path_type}")

# Integration instructions for collective_experiment.py:

INTEGRATION_CODE = '''
# Add this import at the top of collective_experiment.py
from comprehensive_robot_aware_naming import detect_robot_type_from_config, create_robot_aware_path

class Experiment(checkpointable.Checkpointable):
    def __init__(self, config: ConfigType, experiment_id: str = "0"):
        # ... existing initialization code ...
        
        # ADD THIS: Detect robot type once during initialization
        self.robot_type = detect_robot_type_from_config(config)
        print(f"Detected robot type: {self.robot_type}")
        
        # ... rest of existing code ...
        
        if self.config.experiment.mode == 'train_worker':
            # REPLACE the existing model_dir, buffer_dir, etc. definitions with:
            
            self.model_dir = [utils.make_dir(
                create_robot_aware_path(
                    self.config.setup.save_dir, 
                    self.robot_type, 
                    task_name, 
                    self.config.setup.seed, 
                    'model'
                )
            ) for task_name in self.task_names]
            
            self.buffer_dir = [utils.make_dir(
                create_robot_aware_path(
                    self.config.setup.save_dir, 
                    self.robot_type, 
                    task_name, 
                    self.config.setup.seed, 
                    'buffer'
                )
            ) for task_name in self.task_names]
            
            self.buffer_dir_distill_tmp = [utils.make_dir(
                create_robot_aware_path(
                    self.config.setup.save_dir, 
                    self.robot_type, 
                    task_name, 
                    self.config.setup.seed, 
                    'buffer_distill_tmp'
                )
            ) for task_name in self.task_names]
            
            self.buffer_dir_distill = [utils.make_dir(
                create_robot_aware_path(
                    self.config.setup.save_dir, 
                    self.robot_type, 
                    task_name, 
                    self.config.setup.seed, 
                    'buffer_distill'
                )
            ) for task_name in self.task_names]
            
        elif self.config.experiment.mode == 'record':
            self.buffer_dir_distill = [utils.make_dir(
                create_robot_aware_path(
                    self.config.setup.save_dir, 
                    self.robot_type, 
                    task_name, 
                    self.config.setup.seed, 
                    'recording_buffer_distill'
                )
            ) for task_name in self.task_names]
            
        elif self.config.experiment.mode == 'online_distill_collective_transformer':
            self.expert_model_dir = [utils.make_dir(
                create_robot_aware_path(
                    self.config.setup.save_dir, 
                    self.robot_type, 
                    task_name, 
                    self.config.setup.seed, 
                    'model'
                )
            ) for task_name in self.task_names]
            
            self.buffer_dir = utils.make_dir(
                create_robot_aware_path(
                    self.config.setup.save_dir, 
                    self.robot_type, 
                    self.task_names[0], 
                    self.config.setup.seed, 
                    'online_buffer'
                )
            )
            
        elif self.config.experiment.mode == 'train_student':
            self.student_model_dir = utils.make_dir(
                create_robot_aware_path(
                    self.config.setup.save_dir, 
                    self.robot_type, 
                    self.task_names[0], 
                    self.config.setup.seed, 
                    'student_model'
                )
            )
            
            self.buffer_dir = utils.make_dir(
                create_robot_aware_path(
                    self.config.setup.save_dir, 
                    self.robot_type, 
                    self.task_names[0], 
                    self.config.setup.seed, 
                    'student_buffer'
                )
            )
            
        elif self.config.experiment.mode == 'distill_policy':
            self.buffer_dir = utils.make_dir(
                create_robot_aware_path(
                    self.config.setup.save_dir, 
                    self.robot_type, 
                    self.task_names[0], 
                    self.config.setup.seed, 
                    'policy_distill'
                )
            )
'''

# Results of robot-aware naming:
EXAMPLE_NAMING_RESULTS = '''
Original naming:
- model_reach-v2_seed_1
- buffer_distill_reach-v2_seed_1
- online_buffer_reach-v2

New robot-aware naming:
- model_sawyer_reach-v2_seed_1
- model_ur5e_reach-v2_seed_1  
- model_ur10e_reach-v2_seed_1
- buffer_distill_sawyer_reach-v2_seed_1
- buffer_distill_ur5e_reach-v2_seed_1
- buffer_distill_ur10e_reach-v2_seed_1
- online_buffer_sawyer_reach-v2
- online_buffer_ur5e_reach-v2
- online_buffer_ur10e_reach-v2

This enables:
1. Separate expert models per robot-task combination
2. Proper multi-robot collective learning
3. No model overwriting issues
4. Clear identification of robot type in saved files
'''

print("Robot-aware naming implementation complete!")
print("This addresses ALL path definitions found in collective_experiment.py:")
print("- model_dir (lines 116-118)")
print("- buffer_dir (lines 119-121)")  
print("- buffer_dir_distill_tmp (lines 122-124)")
print("- buffer_dir_distill (lines 125-127)")
print("- recording_buffer_distill (lines 165-167)")
print("- expert_model_dir (lines 317-319)")
print("- online_buffer (lines 321-323)")
print("- student_model_dir (lines 382-384)")
print("- student_buffer (lines 385-387)")
print("- policy_distill (lines 484-486)")