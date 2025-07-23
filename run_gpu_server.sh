#!/bin/bash

# GPU Server Optimized Training Script
# Use this script on multi-GPU servers (4x A100 or similar) for faster training

# Set environment variables for optimal GPU performance
export CUDA_VISIBLE_DEVICES=0,1,2,3
export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8

# Configuration settings
GPU_CONFIG="gpu_server_collective_config"
SETUP="gpu_server_metaworld"
ENV_CONFIG="metaworld-mt1"
NUM_ENVS=16

### Utility Functions ###
check_gpu() {
    echo "Checking GPU availability..."
    nvidia-smi
    echo "Number of available GPUs: $(nvidia-smi -L | wc -l)"
}

monitor_gpu() {
    echo "Starting GPU monitoring in background..."
    nvidia-smi -l 5 > gpu_usage.log &
    MONITOR_PID=$!
    echo "GPU monitoring PID: $MONITOR_PID"
}

stop_monitor() {
    if [ ! -z "$MONITOR_PID" ]; then
        kill $MONITOR_PID 2>/dev/null
        echo "Stopped GPU monitoring"
    fi
}

### Training Functions (GPU Optimized) ###
gpu_train_task(){
    local task_name="$1"
    local nr_steps="$2"
    shift 2

    if [[ -z "$task_name" || -z "$nr_steps" ]]; then
        echo "Usage: gpu_train_task <task-name> <nr-steps> [additional args]"
        return 1
    fi

    echo "🚀 GPU Training expert on task: $task_name with $nr_steps steps"
    echo "📊 Additional args: $@"
    
    monitor_gpu
    
    python3 -u main.py \
        --config-name=$GPU_CONFIG \
        setup=$SETUP \
        env=$ENV_CONFIG \
        worker.multitask.num_envs=$NUM_ENVS \
        experiment.mode=train_worker \
        env.benchmark.env_name="${task_name}" \
        experiment.num_train_steps="${nr_steps}" \
        "$@"
        
    stop_monitor
}

gpu_online_distill(){
    local task_name="$1"

    if [[ -z "$task_name" ]]; then
        echo "Usage: gpu_online_distill <task-name>"
        return 1
    fi

    echo "🔄 GPU Online distill: $task_name"
    
    monitor_gpu

    python3 -u main.py \
        --config-name=$GPU_CONFIG \
        setup=$SETUP \
        env=$ENV_CONFIG \
        worker.multitask.num_envs=$NUM_ENVS \
        experiment.mode=online_distill_collective_transformer \
        env.benchmark.env_name="${task_name}"
        
    stop_monitor
}

gpu_train_collective(){
    echo "🤖 Training collective network with GPU optimization..."
    
    monitor_gpu
    
    python3 -u main.py \
        --config-name=$GPU_CONFIG \
        setup=$SETUP \
        env=$ENV_CONFIG \
        worker.multitask.num_envs=$NUM_ENVS \
        experiment.mode=distill_collective_transformer
        
    stop_monitor
}

gpu_train_student(){
    local task_name="$1"
    local nr_steps="$2"
    shift 2

    if [[ -z "$task_name" || -z "$nr_steps" ]]; then
        echo "Usage: gpu_train_student <task-name> <nr-steps> [additional args]"
        return 1
    fi

    echo "🎓 GPU Training student on task: $task_name with $nr_steps steps"
    echo "📊 Additional args: $@"
    
    monitor_gpu

    python3 -u main.py \
        --config-name=$GPU_CONFIG \
        setup=$SETUP \
        env=$ENV_CONFIG \
        worker.multitask.num_envs=$NUM_ENVS \
        experiment.mode=train_student \
        env.benchmark.env_name="${task_name}" \
        experiment.num_student_online_trainsteps2="${nr_steps}" \
        experiment.expert_train_step="${nr_steps}" \
        "$@"
        
    stop_monitor
}

### Evaluation Functions ###
gpu_evaluate_task() {
    local task_name="$1"
    if [[ -z "$task_name" ]]; then
        echo "Usage: gpu_evaluate_task <task-name>"
        return 1
    fi

    echo "📈 GPU Evaluating expert for task: $task_name"

    rm ${PROJECT_ROOT}/logs/experiment_gpu_server/evaluation_models/* 2>/dev/null
    cp ${PROJECT_ROOT}/logs/experiment_gpu_server/model_dir/model_${task_name}_seed_1/* ${PROJECT_ROOT}/logs/experiment_gpu_server/evaluation_models/ 2>/dev/null

    local result_path="${PROJECT_ROOT}/logs/results/gpu_worker/$task_name"
    mkdir -p $(dirname $result_path)

    python3 -u main.py \
        --config-name=$GPU_CONFIG \
        setup=$SETUP \
        env=$ENV_CONFIG \
        worker.multitask.num_envs=$NUM_ENVS \
        experiment.mode=evaluate_collective_transformer \
        env.benchmark.env_name=${task_name} \
        experiment.evaluate_transformer="agent" | tee -a $result_path
}

### Main execution ###
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    echo "🖥️  GPU Server Optimized Training Script"
    echo "========================================"
    
    # Check GPU availability
    check_gpu
    
    # Create result directories
    mkdir -p ${PROJECT_ROOT}/logs/results/gpu_worker
    mkdir -p ${PROJECT_ROOT}/logs/results/gpu_col
    mkdir -p ${PROJECT_ROOT}/logs/results/gpu_student
    
    echo "Starting GPU-optimized training pipeline..."
    
    # Example training pipeline with GPU optimizations
    echo "Phase 1: Training experts with GPU optimization"
    
    # Train key tasks with optimized settings
    gpu_train_task reach-v2 50000    # Reduced steps due to faster training
    gpu_train_task push-v2 400000    # Reduced from 900000
    gpu_train_task pick-place-v2 1200000  # Reduced from 2400000
    
    echo "Phase 1 completed. Check GPU utilization logs in gpu_usage.log"
    
    # Uncomment below for full pipeline
    # gpu_online_distill reach-v2
    # gpu_online_distill push-v2
    # gpu_train_collective
    # gpu_train_student reach-v2 500000
    
else
    # Script was sourced, functions are available
    echo "🔧 GPU optimization functions loaded. Available commands:"
    echo "  - gpu_train_task <task> <steps>"
    echo "  - gpu_online_distill <task>"
    echo "  - gpu_train_collective"
    echo "  - gpu_train_student <task> <steps>"
    echo "  - gpu_evaluate_task <task>"
    echo "  - check_gpu"
    echo "  - monitor_gpu / stop_monitor"
fi