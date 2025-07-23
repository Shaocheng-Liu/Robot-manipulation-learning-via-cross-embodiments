# GPU服务器训练优化完整方案

## 概述
针对你在GTX 1650和4x A100服务器上训练速度相差不大的问题，我创建了一套完整的GPU优化配置方案。这套方案通过调整关键参数来充分利用GPU服务器的计算资源。

## 核心优化策略

### 1. 批次大小 (Batch Size) 优化
- **原始**: 512 → **优化后**: 2048 (4倍提升)
- **影响**: 更好地利用GPU并行计算能力，提高训练吞吐量

### 2. 并行环境数量优化  
- **原始**: 1个环境 → **优化后**: 16个并行环境
- **影响**: 充分利用多核CPU进行环境仿真，增加数据生成速度

### 3. 内存容量优化
- **Replay Buffer**: 140K → 500K
- **Collective Buffer**: 2M → 5M  
- **影响**: 更大的经验池提供更好的样本多样性

### 4. 学习率自适应调整
- **原始**: 3e-4 → **优化后**: 1e-3
- **原理**: 按照batch size增长的平方根调整学习率

### 5. 更新频率优化
- **Actor更新**: 2 → 1 (更频繁)
- **Critic目标更新**: 2 → 1 (更频繁)
- **影响**: 配合更大batch size实现更稳定的训练

## 文件结构
```
config/
├── gpu_server_collective_config.yaml          # 主配置文件
├── setup/gpu_server_metaworld.yaml           # GPU服务器设置
├── experiment/gpu_server_collective_metaworld.yaml  # 实验参数优化
├── replay_buffer/gpu_server_mtrl.yaml        # 缓冲区优化
├── worker/
│   ├── gpu_server_worker_sac.yaml           # Worker SAC优化
│   ├── components/gpu_server_metaworld_multitask.yaml  # 多任务优化
│   └── optimizers/gpu_server_metaworld_*.yaml  # 优化器配置
├── transformer_collective_network/
│   ├── gpu_server_transformer_collective_sac.yaml  # Transformer优化
│   └── optimizers/gpu_server_metaworld_*.yaml      # Transformer优化器
└── student/
    ├── gpu_server_student_sac.yaml          # Student优化
    └── optimizers/gpu_server_metaworld_*.yaml  # Student优化器
```

## 使用方式

### 训练Worker (SAC专家)
```bash
python3 -u main.py \
    --config-name=gpu_server_collective_config \
    setup=gpu_server_metaworld \
    env=metaworld-mt1 \
    worker.multitask.num_envs=16 \
    experiment.mode=train_worker \
    env.benchmark.env_name="reach-v2" \
    experiment.num_train_steps=100000
```

### 训练Collective Network (SAC+Transformer)
```bash  
python3 -u main.py \
    --config-name=gpu_server_collective_config \
    setup=gpu_server_metaworld \
    env=metaworld-mt1 \
    worker.multitask.num_envs=16 \
    experiment.mode=distill_collective_transformer
```

### 训练Student
```bash
python3 -u main.py \
    --config-name=gpu_server_collective_config \
    setup=gpu_server_metaworld \
    env=metaworld-mt1 \
    worker.multitask.num_envs=16 \
    experiment.mode=train_student \
    env.benchmark.env_name="reach-v2" \
    experiment.num_student_online_trainsteps2=500000
```

## 预期性能提升

### 1. Worker训练阶段
- **训练速度**: 3-5倍提升
- **GPU利用率**: 从20-30% → 80-90%
- **内存使用**: 更充分利用GPU显存

### 2. Collective Learning阶段  
- **Transformer训练**: 显著加速（计算密集型）
- **批次处理**: 更高效的梯度计算
- **收敛速度**: 更快达到目标性能

### 3. Student训练阶段
- **知识蒸馏**: 更高效的师生学习
- **策略优化**: 更稳定的策略改进

## 监控和调试

### GPU使用监控
```bash
# 使用提供的脚本自动监控
./run_gpu_server.sh

# 或手动监控
nvidia-smi -l 1
```

### 配置验证
```bash
# 验证配置文件正确性
python3 test_gpu_config.py
```

### 常见问题解决

1. **显存不足**: 减少batch_size到1024
2. **GPU利用率低**: 增加num_envs到24-32
3. **训练不稳定**: 降低学习率到5e-4

## 优化参数对比表

| 参数 | 原始配置 | GPU优化配置 | 提升倍数 |
|------|----------|-------------|----------|
| Batch Size | 512 | 2048 | 4x |
| Num Envs | 1 | 16 | 16x |
| Replay Buffer | 140K | 500K | 3.6x |
| Collective Buffer | 2M | 5M | 2.5x |
| Actor LR | 3e-4 | 1e-3 | 3.3x |
| Actor Update Freq | 2 | 1 | 2x |
| Save Freq | 20K | 10K | 2x |

## 总结

通过这套完整的GPU优化方案，你应该能够在4x A100服务器上获得显著的训练加速。关键是要确保：

1. **充分利用GPU并行计算**: 通过更大的batch size
2. **提高数据生成效率**: 通过更多并行环境  
3. **优化内存使用**: 通过更大的缓冲区容量
4. **保持训练稳定性**: 通过相应调整学习率和更新频率

建议首先用较小的任务（如reach-v2）测试配置，确认GPU利用率达到80%以上后，再进行完整的训练流程。