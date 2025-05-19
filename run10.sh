#python3 -u main.py setup=metaworld env=metaworld-mt1 worker.multitask.num_envs=1 experiment.mode=train_worker env.benchmark.env_name=reach-v2 experiment.num_train_steps=100000
python3 -u main.py setup=metaworld env=metaworld-mt1 worker.multitask.num_envs=1 experiment.mode=train_worker env.benchmark.env_name=push-v2 experiment.num_train_steps=900000
#python3 -u main.py setup=metaworld env=metaworld-mt1 worker.multitask.num_envs=1 experiment.mode=train_worker env.benchmark.env_name=pick-place-v2 experiment.num_train_steps=2400000
python3 -u main.py setup=metaworld env=metaworld-mt1 worker.multitask.num_envs=1 experiment.mode=train_worker env.benchmark.env_name=door-open-v2 experiment.num_train_steps=1000000
#python3 -u main.py setup=metaworld env=metaworld-mt1 worker.multitask.num_envs=1 experiment.mode=train_worker env.benchmark.env_name=drawer-open-v2 experiment.num_train_steps=500000 #(maybe more samples)
python3 -u main.py setup=metaworld env=metaworld-mt1 worker.multitask.num_envs=1 experiment.mode=train_worker env.benchmark.env_name=drawer-close-v2 experiment.num_train_steps=200000
python3 -u main.py setup=metaworld env=metaworld-mt1 worker.multitask.num_envs=1 experiment.mode=train_worker env.benchmark.env_name=button-press-topdown-v2 experiment.num_train_steps=500000 #may need some finetuning
#python3 -u main.py setup=metaworld env=metaworld-mt1 worker.multitask.num_envs=1 experiment.mode=train_worker env.benchmark.env_name=peg-insert-side-v2 experiment.num_train_steps=1300000
python3 -u main.py setup=metaworld env=metaworld-mt1 worker.multitask.num_envs=1 experiment.mode=train_worker env.benchmark.env_name=window-open-v2  experiment.num_train_steps=300000
python3 -u main.py setup=metaworld env=metaworld-mt1 worker.multitask.num_envs=1 experiment.mode=train_worker env.benchmark.env_name=window-close-v2 experiment.num_train_steps=400000 # weired solution 
exit
 





python3 -u main.py setup=metaworld env=metaworld-mt1 worker.multitask.num_envs=1 experiment.mode=online_distill_collective_transformer env.benchmark.env_name=door-open-v2



python split_buffer_files.py   --source logs/experiment_test/buffer/buffer_distill/buffer_distill_push-v2_seed_1   --train  logs/experiment_test/buffer/buffer_distill/train/buffer_distill_push-v2_seed_1   --val    logs/experiment_test/buffer/buffer_distill/validation/buffer_distill_push-v2_seed_1

python split_buffer_files.py   --source logs/experiment_test/buffer/buffer_distill/buffer_distill_reach-v2_seed_1/   --train  logs/experiment_test/buffer/buffer_distill/train/buffer_distill_reach-v2_seed_1   --val    logs/experiment_test/buffer/buffer_distill/validation/buffer_distill_reach-v2_seed_1

python3 Transformer_RNN/dataset_tf.py
python3 Transformer_RNN/RepresentationTransformerWithCLS.py


python split_buffer_files.py   --source logs/experiment_test/buffer/online_buffer_*   --train  logs/experiment_test/buffer/collective_buffer/train/buffer_distill_push-v2_seed_1   --val    logs/experiment_test/buffer/collective_buffer/validation/buffer_distill_push-v2_seed_1

python3 -u main.py setup=metaworld env=metaworld-mt1 worker.multitask.num_envs=1 experiment.mode=distill_collective_transformer


python3 -u main.py setup=metaworld env=metaworld-mt1 worker.multitask.num_envs=1 experiment.mode=evaluate_collective_transformer


tensorboard --logdir logs/



# Evaluate performance
#  rm -r logs/experiment_test/evaluation_models/*
#  cp -r logs/experiment_test/model_dir/model_* logs/experiment_test/evaluation_models/


evaluate_task() {
    local project_root="/home/len1218/documents/BT/framework"
    local task_name="$1"
    if [[ -z "$task_name" ]]; then
        echo "Usage: evaluate_task <task-name>"
        return 1
    fi

    echo "Evaluating task: $task_name"

    rm ${project_root}/logs/experiment_test/evaluation_models/*
    cp ${project_root}/logs/experiment_test/evaluation_models/model_${task_name}_seed_1/* ${project_root}/logs/experiment_test/evaluation_models/

    python3 -u main.py \
        setup=metaworld \
        env=metaworld-mt1 \
        worker.multitask.num_envs=1 \
        experiment.mode=evaluate_collective_transformer \
        env.benchmark.env_name=${task_name} \
        experiment.evaluate_transformer="agent"
}

train_task(){
    local project_root="/home/len1218/documents/BT/framework"
    local task_name="$1"
    local nr_steps="$2"
    if [[ -z "$task_name" || -z "$nr_steps" ]]; then
        echo "Usage: train_task <task-name> <nr-steps>"
        return 1
    fi

    echo "Training expert on task: $task_name"

    python3 -u main.py \
        setup=metaworld \
        env=metaworld-mt1 \
        worker.multitask.num_envs=1 \
        experiment.mode=train_worker \
        env.benchmark.env_name=${task_name} \
        experiment.num_train_steps=${nr_steps}
}

split_buffer(){
    local project_root="/home/len1218/documents/BT/framework"
    local task_name="$1"
    if [[ -z "$task_name" ]]; then
        echo "Usage: split <task-name>"
        return 1
    fi

    echo "Evaluating task: $task_name"
    python split_buffer_files.py \
        --source ${project_root}/logs/experiment_test/buffer/buffer_distill/buffer_distill_${task_name}_seed_1 \
        --train  ${project_root}/logs/experiment_test/buffer/buffer_distill/train/buffer_distill_${task_name}_seed_1 \
        --val    ${project_root}/logs/experiment_test/buffer/buffer_distill/validation/buffer_distill_${task_name}_seed_1
}


train_task window-open-v2

evaluate_task reach-v2
evaluate_task door-open-v2
evaluate_task drawer-close-v2
evaluate_task button-press-topdown-v2
evaluate_task window-open-v2
evaluate_task window-close-v2
