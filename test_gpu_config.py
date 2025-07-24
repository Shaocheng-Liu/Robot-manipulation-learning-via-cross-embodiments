#!/usr/bin/env python3
"""
Test script to verify GPU server optimized configurations work correctly.
This script validates that the configurations load properly and shows the key parameters.
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__)))

try:
    import hydra
    from omegaconf import DictConfig, OmegaConf
    import torch
    
    def test_gpu_config():
        """Test that GPU server config loads correctly"""
        print("🔧 Testing GPU Server Configuration...")
        
        # Initialize Hydra
        with hydra.initialize(config_path="config", version_base=None):
            cfg = hydra.compose(config_name="gpu_server_collective_config", 
                              overrides=["env=metaworld-mt1"])
            
            print("✅ Configuration loaded successfully!")
            
            # Print key optimizations
            print("\n📊 Key GPU Optimizations:")
            print(f"  - Setup: {cfg.setup}")
            print(f"  - Device: {cfg.setup.device}")
            print(f"  - Batch Size: {cfg.replay_buffer.batch_size}")
            print(f"  - Collective Batch Size: {cfg.col_replay_buffer.batch_size}")
            print(f"  - Num Envs: {cfg.worker.multitask.num_envs}")
            print(f"  - Actor LR: {cfg.worker.optimizers.actor.lr}")
            print(f"  - Critic LR: {cfg.worker.optimizers.critic.lr}")
            print(f"  - Actor Update Freq: {cfg.worker.builder.actor_update_freq}")
            print(f"  - Replay Buffer Capacity: {cfg.replay_buffer.capacity}")
            
            # Check GPU availability
            print(f"\n🖥️  GPU Status:")
            print(f"  - CUDA Available: {torch.cuda.is_available()}")
            if torch.cuda.is_available():
                print(f"  - GPU Count: {torch.cuda.device_count()}")
                print(f"  - Current Device: {torch.cuda.current_device()}")
                print(f"  - Device Name: {torch.cuda.get_device_name(0)}")
            
            return True
            
    if __name__ == "__main__":
        try:
            test_gpu_config()
            print("\n✅ All GPU optimization tests passed!")
        except Exception as e:
            print(f"\n❌ Test failed: {e}")
            sys.exit(1)
            
except ImportError as e:
    print(f"❌ Import error: {e}")
    print("Please install required dependencies: pip install hydra-core omegaconf torch")
    sys.exit(1)