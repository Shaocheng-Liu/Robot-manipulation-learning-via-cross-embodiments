
CUDA_VISIBLE_DEVICES=2 python3 -u main.py setup=metaworld env=metaworld-mt1 worker.multitask.num_envs=1 experiment.mode=train_world_model transformer_collective_network.world_model.load_on_init=False

CUDA_VISIBLE_DEVICES=2 python3 -u main.py setup=metaworld env=metaworld-mt1 worker.multitask.num_envs=1 experiment.mode=distill_collective_transformer 
