[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://github.com/facebookresearch/mtrl/blob/main/LICENSE)
[![Python 3.8](https://img.shields.io/badge/python-3.8-blue.svg)](https://www.python.org/downloads/release/python-380/)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Zulip Chat](https://img.shields.io/badge/zulip-join_chat-brightgreen.svg)](https://mtenv.zulipchat.com)

# A transformer based collective learning framework for scalable knowledge accumulation and transfer


## Contents

1. [Introduction](#Introduction)

2. [Setup](#Setup)

3. [Run the code](#Run-the-code)

4. [Acknowledgements](#Acknowledgements)

## Introduction

Reinforcement learning (RL) has achieved remarkable success in addressing complex decision-making problems, particularly through the integration of deep neural networks. Despite these advancements, RL agents face persistent challenges, including sample inefficiency, scalability across multiple tasks, and limited generalization to unseen environments, which hinder their applicability in real-world scenarios. This thesis introduces a novel framework that combines a transformer-based encoder with a collective learning strategy to address these limitations.
The proposed approach leverages transformers to process sequential trajectory data, enabling the policy network to capture rich contextual information. A collective learning framework is employed, wherein multiple RL agents share their experiences to train a centralized policy network. This collaborative process reduces sample requirements, enhances scalability for multi-task learning, and improves generalization across diverse environments. Furthermore, a reward-shaping strategy is introduced to utilize the knowledge of the collective network, accelerating the learning of new policies while avoiding convergence to suboptimal solutions. 
Experimental evaluations on the MetaWorld benchmark demonstrate that the proposed framework achieves superior sample efficiency, scalability, and generalization compared to single-task policies. Additionally, the collective network shows strong potential for lifelong learning by continually acquiring and adapting to new skills. The framework is also validated for cross-embodiment policy transfer, successfully generalizing control policies to robotic arms with varying morphologies. 

### Demonstration
![img](/imgs/snapshot.png "Snapshot")

### Snapshots
[![Movie1](/imgs/panda_drawer_120.png "Demonstration video")](https://youtu.be/5edG_Wm39Mc)

## Setup
A full installation requires python 3.8:
* Clone the repository: `git clone https://github.com/4nd1L0renz/A-transformer-based-collective-learning-framework-for-scalable-knowledge-accumulation-and-transfer.git`.

* Install dependencies: `pip install -r requirements/dev.txt`
  
* Install modified Metaworld `cd Metaworld & pip install -e .`
    
* Note that we use mtenv to manage Meta-World environment, and add slight modification under **mtenv/envs/metaworld/env.py**, we added following function allowing for output a list of env instances:
```
def get_list_of_envs(
    benchmark: Optional[metaworld.Benchmark],
    benchmark_name: str,
    env_id_to_task_map: Optional[EnvIdToTaskMapType],
    should_perform_reward_normalization: bool = True,
    task_name: str = "pick-place-v1",
    num_copies_per_env: int = 1,
) -> Tuple[List[Any], Dict[str, Any]]:

    if not benchmark:
        if benchmark_name == "MT1":
            benchmark = metaworld.ML1(task_name)
        elif benchmark_name == "MT10":
            benchmark = metaworld.MT10()
        elif benchmark_name == "MT50":
            benchmark = metaworld.MT50()
        else:
            raise ValueError(f"benchmark_name={benchmark_name} is not valid.")

    env_id_list = list(benchmark.train_classes.keys())

    def _get_class_items(current_benchmark):
        return current_benchmark.train_classes.items()

    def _get_tasks(current_benchmark):
        return current_benchmark.train_tasks

    def _get_env_id_to_task_map() -> EnvIdToTaskMapType:
        env_id_to_task_map: EnvIdToTaskMapType = {}
        current_benchmark = benchmark
        for env_id in env_id_list:
            for name, _ in _get_class_items(current_benchmark):
                if name == env_id:
                    task = random.choice(
                        [
                            task
                            for task in _get_tasks(current_benchmark)
                            if task.env_name == name
                        ]
                    )
                    env_id_to_task_map[env_id] = task
        return env_id_to_task_map

    if env_id_to_task_map is None:
        env_id_to_task_map: EnvIdToTaskMapType = _get_env_id_to_task_map()  # type: ignore[no-redef]
    assert env_id_to_task_map is not None

    def make_envs_use_id(env_id: str):
        current_benchmark = benchmark
        
        
        def _make_env():
            for name, env_cls in _get_class_items(current_benchmark):
                if name == env_id:
                    env = env_cls()
                    task = env_id_to_task_map[env_id]
                    env.set_task(task)
                    if should_perform_reward_normalization:
                        env = NormalizedEnvWrapper(env, normalize_reward=True)
                    return env
        # modified return built single envs
        single_env = _make_env()
        return single_env

    if num_copies_per_env > 1:
        env_id_list = [
            [env_id for _ in range(num_copies_per_env)] for env_id in env_id_list
        ]
        env_id_list = [
            env_id for env_id_sublist in env_id_list for env_id in env_id_sublist
        ]

    list_of_envs = [make_envs_use_id(env_id) for env_id in env_id_list]
    return list_of_envs, env_id_to_task_map
```

These are the steps all in one script:
```bash
git clone git@github.com:argator18/A-transformer-based-collective-learning-framework-for-scalable-knowledge-accumulation-and-transfer.git framework
cd framework

# install python 3.8
pyenv install 3.8 
 ~/.pyenv/versions/3.8.20/bin/python3 -m venv venv

# set env variable to set project root
echo "export PROJECT_ROOT=$(pwd)" >> venv/bin/activate
source venv/bin/activate

pip install -r requirements/dev.txt
pip install mujoco==3.2.3

git clone git@github.com:facebookresearch/mtenv.git
cd mtenv

# apply custom patches
patch -p1 < ../changes.diff

pip install -e .

pip uninstall gym
pip install gym==0.26.0
pip install gymnasium==1.0.0a2

pip install -e .[all]

cd ../Metaworld
pip install -e .
pip install gym==0.21.0
pip install protobuf==3.20.3
pip install numpy==1.23.5
pip install bnpy

cd ..
pip install -e .
```
## Run the code

### Config

Our code uses Hydra to manage the configuration for the experiments. The config can be found in the `config`-folder. The most important settings to check before running the code is:

* experiment/collective_metaworld.yaml: Specify the training mode and the details for training; supported modes are: train_worker/online_distill_collective_transformer/distill_collective_transformer/evaluate_collective_transformer/train_student/train_student_finetuning/record

* env/metaworld-mt1.yaml or env/metaworld-mt10.yaml: To select the tasks.

* The remaining experiment settings and the hyperparameter of the model can be found in their respective folder in the config-folder

### Execution
A full example on how to run the code can be found in `run.sh`. It provides for most tasks the perfect hyperparameters and bash wrapper to execute single tasks more convenient. Additionally it shows in which files the input and outputs are expected.  

1. **Experiment mode Train expert**: Train an expert on a task. While training collect regular experience samples for later training of the transformer trajectory encoder

2. **Experiment mode Online Distill collective network**: Creates the offline dataset for training the collective network by distilling the expert knowledge into a temporary network while also recording state, action, rewards etc

3. **Train trajectory transformer**: Create torch dataset by running `Transformer_RNN/dataset_tf.py` (specify location of training data in code); Run training via `Transformer_RNN/RepresentationTransformerWithCLS.py`

4. **Experiment mode Distill collective network**: Loads all collected datasets and runs the training of the collective network; expects the training data in `(experimentfolder)/buffer/collective_buffer/train` and validation data in `(experimentfolder)/buffer/collective_buffer/validation`

5. **Experiment mode Evaluate_collective_transformer**: Evaluates the performance of experts or collective network

6. **Experiment mode Train_student**: Loads the collective network and teaches a student policy a learned task by providing it an addtional reward

7. **Experiment mode Train_student_finetuning**: Loads the collective network and finetunes it on a new task in standard SAC manner

* **Experiment mode Record**: Used only for test purposes. Uses either random-, scripted or probabilistic scripted policy to recorde experiences

* **Different embodiments**: For changing between different robots (Kuka, Sawyer, Panda) please adjust `Metaworld/metaworld/envs/assets_v2/sawyer_xyz/(environment name).xml` and uncomment the robot type you want. For the Panda robot please also additionally uncomment line 689-691 in `Metaworld/metaworld/envs/mujoco/sawyer_xyz/saywer_xyz_env.py`

After setting the configuration for the experiment run the code as follows (for metaworld-mt1):

**Standard Training (single GPU)**:
```
python3 -u main.py setup=metaworld env=metaworld-mt1 worker.multitask.num_envs=1
```

**GPU Server Optimized Training (multi-GPU servers like 4x A100)**:
```
python3 -u main.py --config-name=gpu_server_collective_config setup=gpu_server_metaworld env=metaworld-mt1 worker.multitask.num_envs=16
```

For detailed GPU optimization instructions, see [GPU_OPTIMIZATION_GUIDE.md](GPU_OPTIMIZATION_GUIDE.md).


## Acknowledgements

* Project file pre-commit, mypy config, towncrier config, circleci etc are based on same files from [Hydra](https://github.com/facebookresearch/hydra).

* Implementation Inherited from [MTRL](https://mtrl.readthedocs.io/en/latest/index.html) library. 

* Documentation of MTRL repository refer to: [https://mtrl.readthedocs.io](https://mtrl.readthedocs.io).
