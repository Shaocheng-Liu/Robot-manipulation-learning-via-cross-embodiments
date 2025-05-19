# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

"""This is the main entry point for the code."""

import hydra

from mtrl.app.run import run
from mtrl.utils import config as config_utils
from mtrl.utils.types import ConfigType

######################################################################

import torch, numpy as np, builtins
def log_blue(msg):
    print(f"\033[34m{msg}\033[0m")

log_blue("[SAVE] Replay buffer to ./save/path")

_orig_save = torch.save
def save_with_log(obj, f, *args, **kwargs):
    log_blue(f"[TORCH SAVE] → {f}")
    return _orig_save(obj, f, *args, **kwargs)
torch.save = save_with_log

_orig_load = torch.load
def load_with_log(f, *args, **kwargs):
    log_blue(f"[TORCH LOAD] ← {f}")
    return _orig_load(f, *args, **kwargs)
torch.load = load_with_log

# Similarly for np.save / np.load
np_save = np.save
np_load = np.load

def logged_np_save(file, arr, *args, **kwargs):
    log_blue(f"[NP SAVE] → {file}")
    return np_save(file, arr, *args, **kwargs)
np.save = logged_np_save

def logged_np_load(file, *args, **kwargs):
    log_blue(f"[NP LOAD] ← {file}")
    return np_load(file, *args, **kwargs)
np.load = logged_np_load


######################################################################



@hydra.main(config_path="config", config_name="collective_config")
def launch(config: ConfigType) -> None:
    config = config_utils.process_config(config)
    return run(config)


if __name__ == "__main__":
    launch()
