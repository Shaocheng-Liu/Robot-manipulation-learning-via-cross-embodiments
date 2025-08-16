python3 Transformer_RNN/dataset_tf.py

mv Transformer_RNN/decision_tf_dataset/_chunk_0 Transformer_RNN/decision_tf_dataset/train/_chunk_0
mv Transformer_RNN/decision_tf_dataset/_chunk_1 Transformer_RNN/decision_tf_dataset/validation/_chunk_0

python3 Transformer_RNN/RepresentationTransformerWithCLS.py

# train col network
python3 -u main.py setup=metaworld env=metaworld-mt1 worker.multitask.num_envs=1 experiment.mode=distill_collective_transformer 

mv Transformer_RNN/checkpoints temp/checkpoints_tt_2

mv logs/experiment_test/model_dir/model_col temp/model_col_tt_push_changed

python3 -u main.py setup=metaworld env=metaworld-mt1 worker.multitask.num_envs=1 experiment.mode=distill_collective_transformer


