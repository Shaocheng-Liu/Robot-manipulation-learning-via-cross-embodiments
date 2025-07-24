# GPU Server Optimization Guide

本文档详细说明了如何在GPU服务器（如4x A100）上优化机器人操控学习训练的参数配置。

## 快速开始

### 1. Worker训练 (SAC)
```bash
# 使用GPU优化配置训练单任务专家
python3 -u main.py \
    --config-name=gpu_server_collective_config \
    setup=gpu_server_metaworld \
    env=metaworld-mt1 \
    worker.multitask.num_envs=16 \
    experiment.mode=train_worker \
    env.benchmark.env_name="reach-v2" \
    experiment.num_train_steps=100000
```

### 2. Collective Learning (SAC+Transformer)
```bash
# 训练集体网络
python3 -u main.py \
    --config-name=gpu_server_collective_config \
    setup=gpu_server_metaworld \
    env=metaworld-mt1 \
    worker.multitask.num_envs=16 \
    experiment.mode=distill_collective_transformer
```

### 3. Student训练
```bash
# 使用集体网络训练学生策略
python3 -u main.py \
    --config-name=gpu_server_collective_config \
    setup=gpu_server_metaworld \
    env=metaworld-mt1 \
    worker.multitask.num_envs=16 \
    experiment.mode=train_student \
    env.benchmark.env_name="reach-v2" \
    experiment.num_student_online_trainsteps2=500000
```

## 主要优化项目

### 1. 批次大小优化 (Batch Size)
- **原始配置**: 512
- **GPU服务器配置**: 2048
- **原理**: 更大的批次能更好地利用GPU计算资源，提高训练吞吐量

### 2. 并行环境数量 (Parallel Environments)
- **原始配置**: 1个环境
- **GPU服务器配置**: 16个并行环境
- **原理**: 多个并行环境可以充分利用多核CPU和GPU的并行计算能力

### 3. 缓冲区容量 (Buffer Capacity)
- **原始replay buffer**: 140,000 → 500,000
- **原始collective buffer**: 2,000,000 → 5,000,000  
- **原理**: 更大的缓冲区提供更多样化的经验，提高训练稳定性

### 4. 学习率调整 (Learning Rate)
- **原始配置**: 3e-4
- **GPU服务器配置**: 1e-3
- **原理**: 由于批次大小增加了4倍，学习率按sqrt(batch_size_ratio)≈2倍调整

### 5. 更新频率优化 (Update Frequency)
- **Actor更新频率**: 2 → 1 (更频繁的更新)
- **Critic目标网络更新**: 2 → 1 (更频繁的目标更新)
- **原理**: 更大的批次允许更频繁的参数更新，加速收敛

### 6. 采样和保存频率优化
- **集体采样频率**: 10,000 → 5,000 steps
- **评估频率**: 4,000 → 2,000 steps  
- **保存频率**: 20,000 → 10,000 steps
- **原理**: 更频繁的采样和评估，更好地监控训练进度

## 不同训练阶段的配置建议

### Stage 1: Worker Training (专家训练)
**推荐参数**:
```yaml
# 在gpu_server_collective_config.yaml中设置
experiment.mode: train_worker
worker.multitask.num_envs: 16
replay_buffer.batch_size: 2048
experiment.col_sampling_freq: 5000
experiment.save_freq: 10000
```

**预期提升**: 相比单GPU训练，速度提升3-5倍

### Stage 2: Collective Learning (集体学习)
**推荐参数**:
```yaml
experiment.mode: distill_collective_transformer  
transformer_collective_network.builder.actor_update_freq: 1
transformer_collective_network.builder.critic_target_update_freq: 1
col_replay_buffer.batch_size: 2048
experiment.num_actor_train_step: 50001
```

**预期提升**: 由于Transformer计算密集，GPU利用率显著提升

### Stage 3: Student Training (学生训练)
**推荐参数**:
```yaml
experiment.mode: train_student
student.builder.actor_update_freq: 1
student.builder.critic_target_update_freq: 1
experiment.init_steps_stu2: 2000  # 减少预热步数
```

**预期提升**: 结合集体网络的知识，收敛更快

## 内存和计算优化

### GPU内存使用
- **4x A100 (80GB each)**: 可以支持更大的模型和批次
- **建议监控**: 使用`nvidia-smi`监控GPU利用率，目标>85%

### CPU并行处理
- **数据加载workers**: 8个worker进程
- **环境并行**: 16个并行MetaWorld环境
- **建议**: 确保CPU核心数≥32

### 自动混合精度 (AMP)
```yaml
gpu_optimizations:
  use_amp: true  # 启用AMP，节省显存，加速训练
```

## 故障排除

### 1. 显存不足 (OOM)
- 减少batch_size到1024
- 减少num_envs到8
- 启用gradient checkpointing

### 2. GPU利用率低
- 增加batch_size
- 增加num_envs
- 检查数据加载瓶颈

### 3. 训练不稳定
- 降低学习率到5e-4
- 增加warmup steps
- 检查梯度裁剪设置

## 性能监控

### 关键指标
1. **GPU利用率**: 目标>80%
2. **内存使用**: 目标<90%
3. **训练步数/秒**: 相比单GPU提升3-5倍
4. **收敛速度**: 更快达到目标性能

### 监控命令
```bash
# GPU监控
nvidia-smi -l 1

# 训练日志监控  
tail -f logs/experiment_gpu_server/train.log

# TensorBoard监控
tensorboard --logdir=logs/experiment_gpu_server
```

## 高级优化

### 1. 分布式训练
```yaml
# 未来版本支持
distributed:
  backend: nccl
  world_size: 4
  rank: 0
```

### 2. 模型编译 (PyTorch 2.0+)
```yaml
gpu_optimization:
  compile_model: true  # 需要PyTorch >= 2.0
```

### 3. 自定义CUDA内核
对于特定操作，可以考虑自定义CUDA实现以进一步提升性能。

## 总结

通过这些GPU优化配置，您应该能够在4x A100服务器上获得相比单GPU 3-5倍的训练速度提升。关键是要平衡批次大小、并行环境数量和学习率，以充分利用GPU计算资源。