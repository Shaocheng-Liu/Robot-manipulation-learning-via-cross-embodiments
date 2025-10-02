
CUDA_VISIBLE_DEVICES=2 python3 -u main.py setup=metaworld env=metaworld-mt1 worker.multitask.num_envs=1 experiment.mode=train_world_model transformer_collective_network.world_model.load_on_init=False

CUDA_VISIBLE_DEVICES=2 python3 -u main.py setup=metaworld env=metaworld-mt1 worker.multitask.num_envs=1 experiment.mode=distill_collective_transformer 

mv logs/experiment_test/model_dir/model_col logs/experiment_test/model_dir/model_col_seed_3_no_tt_wm_norm

mv logs/experiment_test/model_dir/model_world logs/experiment_test/model_dir/model_world_200000_seed_3

mv Transformer_RNN/checkpoints_seed_3 Transformer_RNN/checkpoints

CUDA_VISIBLE_DEVICES=2 python3 -u main.py setup=metaworld env=metaworld-mt1 worker.multitask.num_envs=1 experiment.mode=train_world_model transformer_collective_network.world_model.load_on_init=False

CUDA_VISIBLE_DEVICES=2 python3 -u main.py setup=metaworld env=metaworld-mt1 worker.multitask.num_envs=1 experiment.mode=distill_collective_transformer 

mv logs/experiment_test/model_dir/model_col logs/experiment_test/model_dir/model_col_seed_3_tt_wm_norm

mv logs/experiment_test/model_dir/model_world logs/experiment_test/model_dir/model_world_200000_seed_3_TT