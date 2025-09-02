CUDA_VISIBLE_DEVICES=2 python3 Transformer_RNN/RepresentationTransformerWithCLS.py

CUDA_VISIBLE_DEVICES=2 python3 -u main.py setup=metaworld env=metaworld-mt1 worker.multitask.num_envs=1 experiment.mode=distill_collective_transformer 