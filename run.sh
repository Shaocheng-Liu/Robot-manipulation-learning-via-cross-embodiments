python3 -u main.py setup=metaworld env=metaworld-mt1 worker.multitask.num_envs=1 experiment.mode=train_worker env.benchmark.env_name=push-v2
python3 -u main.py setup=metaworld env=metaworld-mt1 worker.multitask.num_envs=1 experiment.mode=train_worker env.benchmark.env_name=reach-v2
python3 -u main.py setup=metaworld env=metaworld-mt1 worker.multitask.num_envs=1 experiment.mode=online_distill_collective_transformer

python3 Transformer_RNN/dataset_tf.py
python3 Transformer_RNN/RepresentationTransformerWithCLS.py


python3 -u main.py setup=metaworld env=metaworld-mt1 worker.multitask.num_envs=1 experiment.mode=distill_collective_transformer
