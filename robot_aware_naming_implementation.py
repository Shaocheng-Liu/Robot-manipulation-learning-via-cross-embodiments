#!/usr/bin/env python3
"""
Robot-Aware Model Naming Implementation

This script demonstrates how to modify the codebase to save separate expert models 
for different robots performing the same task.

Example modification for mtrl/experiment/collective_learning.py
"""

import os
from pathlib import Path
from typing import Optional


class RobotAwareModelNaming:
    """
    Implementation example for robot-aware model naming system.
    
    This should be integrated into the existing collective_learning.py
    """
    
    def __init__(self, config):
        self.config = config
        self.robot_type = self.detect_robot_type()
    
    def detect_robot_type(self) -> str:
        """
        Detect robot type from environment configuration or asset files.
        
        Returns:
            Robot type string ('sawyer', 'ur5e', 'ur10e', etc.)
        """
        # Method 1: Check environment assets
        if hasattr(self.config, 'env') and hasattr(self.config.env, 'benchmark'):
            # Look for robot-specific indicators in benchmark config
            env_name = getattr(self.config.env.benchmark, 'env_name', '')
            
            # Check if specific robot is mentioned in config
            robot_indicators = {
                'ur5e': ['ur5e', 'ur_5e'],
                'ur10e': ['ur10e', 'ur_10e'], 
                'sawyer': ['sawyer', 'default']  # sawyer as default
            }
            
            for robot, indicators in robot_indicators.items():
                if any(indicator in env_name.lower() for indicator in indicators):
                    return robot
        
        # Method 2: Check asset files being used
        asset_path = self._get_current_asset_path()
        if asset_path:
            if 'ur5e' in asset_path:
                return 'ur5e'
            elif 'ur10e' in asset_path:
                return 'ur10e'
            elif 'sawyer' in asset_path:
                return 'sawyer'
        
        # Method 3: Check environment class name
        if hasattr(self.config, 'env_class_name'):
            class_name = self.config.env_class_name.lower()
            if 'ur5e' in class_name:
                return 'ur5e'
            elif 'ur10e' in class_name:
                return 'ur10e'
        
        # Default to sawyer if cannot detect
        return 'sawyer'
    
    def _get_current_asset_path(self) -> Optional[str]:
        """Get the current asset path being used by the environment."""
        # This would need to be implemented based on how assets are loaded
        # For now, return None as placeholder
        return None
    
    def get_robot_aware_model_path(self, task_name: str, seed: int = 1) -> str:
        """
        Generate robot-aware model path.
        
        Args:
            task_name: Name of the task (e.g., 'reach-v2')
            seed: Random seed number
            
        Returns:
            Robot-specific model path
        """
        base_dir = self.config.experiment.save_dir
        model_dir = f"model_{self.robot_type}_{task_name}_seed_{seed}"
        return os.path.join(base_dir, "model_dir", model_dir)
    
    def get_robot_aware_buffer_path(self, task_name: str, seed: int = 1, 
                                   buffer_type: str = "buffer") -> str:
        """
        Generate robot-aware buffer path.
        
        Args:
            task_name: Name of the task
            seed: Random seed number
            buffer_type: Type of buffer ('buffer', 'buffer_distill', 'buffer_distill_tmp')
            
        Returns:
            Robot-specific buffer path
        """
        base_dir = self.config.experiment.save_dir
        buffer_dir = f"{buffer_type}_{self.robot_type}_{task_name}_seed_{seed}"
        return os.path.join(base_dir, "buffer", buffer_type, buffer_dir)


def demonstrate_robot_naming():
    """Demonstrate the robot-aware naming system."""
    
    # Simulate different robot configurations
    robot_configs = [
        {'robot_hint': 'sawyer', 'env_name': 'reach-v2'},
        {'robot_hint': 'ur5e', 'env_name': 'reach-v2'},
        {'robot_hint': 'ur10e', 'env_name': 'reach-v2'},
    ]
    
    print("🤖 Robot-Aware Model Naming Demonstration")
    print("=" * 50)
    
    for i, config in enumerate(robot_configs):
        print(f"\nConfiguration {i+1}: {config['robot_hint'].upper()} robot")
        
        # Mock config object
        class MockConfig:
            def __init__(self, robot_hint):
                self.robot_hint = robot_hint
                class Env:
                    class Benchmark:
                        env_name = f"{robot_hint}_reach-v2"  # Include robot hint
                    benchmark = Benchmark()
                self.env = Env()
                
                class Experiment:
                    save_dir = "./logs/experiment_test"
                self.experiment = Experiment()
        
        # Create naming system
        mock_config = MockConfig(config['robot_hint'])
        naming = RobotAwareModelNaming(mock_config)
        
        # Generate paths
        task_name = "reach-v2"
        model_path = naming.get_robot_aware_model_path(task_name)
        buffer_path = naming.get_robot_aware_buffer_path(task_name)
        distill_path = naming.get_robot_aware_buffer_path(task_name, buffer_type="buffer_distill")
        
        print(f"  Robot Type: {naming.robot_type}")
        print(f"  Model Path: {model_path}")
        print(f"  Buffer Path: {buffer_path}")
        print(f"  Distill Path: {distill_path}")


def integration_instructions():
    """Print integration instructions."""
    
    instructions = """
🔧 Integration Instructions

To implement robot-aware model naming in your codebase:

1. **Modify mtrl/experiment/collective_learning.py**:
   
   Add the RobotAwareModelNaming class (above) to the file.
   
   In the __init__ method:
   ```python
   def __init__(self, config: ConfigType, experiment_id: str = "0"):
       super().__init__(config, experiment_id)
       self.robot_naming = RobotAwareModelNaming(config)
   ```

2. **Update model saving logic**:
   
   Find the model saving methods and replace:
   ```python
   # OLD
   model_path = f"model_{task_name}_seed_{seed}"
   
   # NEW  
   model_path = self.robot_naming.get_robot_aware_model_path(task_name, seed)
   ```

3. **Update buffer saving logic**:
   
   Replace buffer path generation:
   ```python
   # OLD
   buffer_path = f"buffer_{task_name}_seed_{seed}"
   
   # NEW
   buffer_path = self.robot_naming.get_robot_aware_buffer_path(task_name, seed)
   ```

4. **Update run.sh functions**:
   
   Modify the bash functions to include robot type:
   ```bash
   # Function to remove robot-specific models
   rm_robot_model(){
       local robot_type="$1"
       local task_name="$2"
       echo "Removing ${robot_type} model for task: $task_name"
       rm logs/experiment_test/model_dir/model_${robot_type}_${task_name}_seed_1/*
   }
   ```

5. **Environment Configuration**:
   
   Ensure your environment config includes robot type identification:
   ```yaml
   env:
     benchmark:
       robot_type: "ur5e"  # or "ur10e", "sawyer"
       env_name: "reach-v2"
   ```

6. **Testing**:
   
   Test with different robots:
   ```bash
   # Train experts for different robots on same task
   train_task reach-v2 100000 env.benchmark.robot_type=sawyer
   train_task reach-v2 100000 env.benchmark.robot_type=ur5e
   train_task reach-v2 100000 env.benchmark.robot_type=ur10e
   ```

Expected Results:
- model_sawyer_reach-v2_seed_1/
- model_ur5e_reach-v2_seed_1/  
- model_ur10e_reach-v2_seed_1/

This enables training multiple robot experts per task for collective learning!
"""
    
    print(instructions)


if __name__ == "__main__":
    demonstrate_robot_naming()
    print("\n" + "=" * 70 + "\n")
    integration_instructions()