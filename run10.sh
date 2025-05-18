#python3 -u main.py setup=metaworld env=metaworld-mt1 worker.multitask.num_envs=1 experiment.mode=train_worker env.benchmark.env_name=reach-v2
python3 -u main.py setup=metaworld env=metaworld-mt1 worker.multitask.num_envs=1 experiment.mode=train_worker env.benchmark.env_name=push-v2
exit
python3 -u main.py setup=metaworld env=metaworld-mt1 worker.multitask.num_envs=1 experiment.mode=train_worker env.benchmark.env_name=pick-place-v2
python3 -u main.py setup=metaworld env=metaworld-mt1 worker.multitask.num_envs=1 experiment.mode=train_worker env.benchmark.env_name=door-open-v2
python3 -u main.py setup=metaworld env=metaworld-mt1 worker.multitask.num_envs=1 experiment.mode=train_worker env.benchmark.env_name=drawer-open-v2 #(maybe more samples)
python3 -u main.py setup=metaworld env=metaworld-mt1 worker.multitask.num_envs=1 experiment.mode=train_worker env.benchmark.env_name=drawer-close-v2
python3 -u main.py setup=metaworld env=metaworld-mt1 worker.multitask.num_envs=1 experiment.mode=train_worker env.benchmark.env_name=button-press-topdown-v2 #may need some finetuning
python3 -u main.py setup=metaworld env=metaworld-mt1 worker.multitask.num_envs=1 experiment.mode=train_worker env.benchmark.env_name=peg-insert-side-v2
python3 -u main.py setup=metaworld env=metaworld-mt1 worker.multitask.num_envs=1 experiment.mode=train_worker env.benchmark.env_name=window-open-v2 
python3 -u main.py setup=metaworld env=metaworld-mt1 worker.multitask.num_envs=1 experiment.mode=train_worker env.benchmark.env_name=window-close-v2 # weired solution
 





python3 -u main.py setup=metaworld env=metaworld-mt1 worker.multitask.num_envs=1 experiment.mode=online_distill_collective_transformer



python split_buffer_files.py   --source logs/experiment_test/buffer/buffer_distill/buffer_distill_push-v2_seed_1   --train  logs/experiment_test/buffer/buffer_distill/train/buffer_distill_push-v2_seed_1   --val    logs/experiment_test/buffer/buffer_distill/validation/buffer_distill_push-v2_seed_1

python split_buffer_files.py   --source logs/experiment_test/buffer/buffer_distill/buffer_distill_reach-v2_seed_1/   --train  logs/experiment_test/buffer/buffer_distill/train/buffer_distill_reach-v2_seed_1   --val    logs/experiment_test/buffer/buffer_distill/validation/buffer_distill_reach-v2_seed_1

python3 Transformer_RNN/dataset_tf.py
python3 Transformer_RNN/RepresentationTransformerWithCLS.py


python split_buffer_files.py   --source logs/experiment_test/buffer/online_buffer_*   --train  logs/experiment_test/buffer/collective_buffer/train/buffer_distill_push-v2_seed_1   --val    logs/experiment_test/buffer/collective_buffer/validation/buffer_distill_push-v2_seed_1

python3 -u main.py setup=metaworld env=metaworld-mt1 worker.multitask.num_envs=1 experiment.mode=distill_collective_transformer


python3 -u main.py setup=metaworld env=metaworld-mt1 worker.multitask.num_envs=1 experiment.mode=evaluate_collective_transformer


tensorboard --logdir logs/
