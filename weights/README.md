This folder holds some pretrained weights.
If you want to use them copy them into `logs/experiment_test/model_dir/`

For evaluation the model has to be in `logs/experiment_test/evaluation_models`. 
Models are copied into this folder automatically by the bash methods in `run.sh`.


### Model overview:
Expert model: 
`model_${task_name}_seed_1/`

Collective network:
`model_col/`


Student model:
`student_model_${task_name}_seed_`

model to evaluate:
`logs/experiment_test/evaluation_models`


