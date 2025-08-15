online_distill(){
    local robot_type="$1"
    local task_name="$2"
    shift 2  # Remove the first two arguments from the list

    if [[ -z "$task_name" ]]; then
        echo "Usage: $0 <task-name>"
        return 1
    fi

    echo "Distill online: $task_name"
    echo "Additional args: $@"
    
    python3 -u main.py \
        setup=metaworld \
        env=metaworld-mt1 \
        worker.multitask.num_envs=1 \
        experiment.mode=online_distill_collective_transformer \
        env.benchmark.env_name="${task_name}" \
        experiment.robot_type="${robot_type}" \
        "$@"
}

online_distill sawyer reach-v2
online_distill sawyer push-v2
online_distill sawyer pick-place-v2
online_distill sawyer door-open-v2
online_distill sawyer drawer-open-v2
online_distill sawyer button-press-v2
online_distill sawyer button-press-topdown-v2
online_distill sawyer peg-insert-side-v2
online_distill sawyer window-open-v2
online_distill sawyer window-close-v2

# online_distill ur10e reach-v2
# online_distill ur10e push-v2
# online_distill ur10e pick-place-v2
# online_distill ur10e door-open-v2
# online_distill ur10e drawer-open-v2
# online_distill ur10e button-press-v2
# online_distill ur10e button-press-topdown-v2
# online_distill ur10e peg-insert-side-v2
# online_distill ur10e window-open-v2
# online_distill ur10e window-close-v2