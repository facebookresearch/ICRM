# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
A command launcher launches a list of commands on a cluster; implement your own
launcher to add support for your cluster. We've provided an example launcher
which runs all commands serially on the local machine.
"""

import subprocess
import time
import torch
import os
import uuid

SLURM_TEMPLATE = """#!/bin/bash
#SBATCH --nodes=1
#SBATCH --gres=gpu:2
#SBATCH --time=48:00:00
#SBATCH --constraint=volta32gb
#SBATCH --job-name=tinyimg
#SBATCH --output=OUTPUT_PATH_PLACEHOLDER/slurm-%j.out
#SBATCH --error=OUTPUT_PATH_PLACEHOLDER/slurm-%j.err

COMMAND_PLACEHOLDER
"""

def multi_node_multi_gpu_launcher(commands, paths):
    for cmd, path in zip(commands, paths):
        # Replace the placeholder in the SLURM script with the actual command
        slurm_script = SLURM_TEMPLATE.replace("COMMAND_PLACEHOLDER", cmd)
        slurm_script = slurm_script.replace("OUTPUT_PATH_PLACEHOLDER", path)

        # Write the modified script to a temporary file. You can use a unique name for each script.
        script_name = f"{path}/slurm_script_{uuid.uuid4().hex}.sh"
        with open(script_name, "w") as f:
            f.write(slurm_script)
        
        # Submit the SLURM job using sbatch
        subprocess.call(["sbatch", script_name])
        
        # Optional: Sleep for a short duration to prevent overwhelming the SLURM scheduler
        time.sleep(2)
        
        
        
def local_launcher(commands, paths):
    """Launch commands serially on the local machine."""
    for cmd in commands:
        subprocess.call(cmd, shell=True)

def dummy_launcher(commands, paths):
    """
    Doesn't run anything; instead, prints each command.
    Useful for testing.
    """
    for cmd in commands:
        print(f'Dummy launcher: {cmd}')

def multi_gpu_launcher(commands, paths):
    """
    Launch commands on the local machine, using all GPUs in parallel.
    """
    print('WARNING: using experimental multi_gpu_launcher.')
    try:
        # Get list of GPUs from env, split by ',' and remove empty string ''
        # To handle the case when there is one extra comma: `CUDA_VISIBLE_DEVICES=0,1,2,3, python3 ...`
        available_gpus = [x for x in os.environ['CUDA_VISIBLE_DEVICES'].split(',') if x != '']
    except Exception:
        # If the env variable is not set, we use all GPUs
        available_gpus = [str(x) for x in range(torch.cuda.device_count())]
    n_gpus = len(available_gpus)
    procs_by_gpu = [None]*n_gpus

    while len(commands) > 0:
        for idx, gpu_idx in enumerate(available_gpus):
            proc = procs_by_gpu[idx]
            if (proc is None) or (proc.poll() is not None):
                # Nothing is running on this GPU; launch a command.
                cmd = commands.pop(0)
                new_proc = subprocess.Popen(
                    f'CUDA_VISIBLE_DEVICES={gpu_idx} {cmd}', shell=True)
                procs_by_gpu[idx] = new_proc
                break
        time.sleep(1)

    # Wait for the last few tasks to finish before returning
    for p in procs_by_gpu:
        if p is not None:
            p.wait()

REGISTRY = {
    'local': local_launcher,
    'dummy': dummy_launcher,
    'multi_gpu': multi_gpu_launcher,
    'multi_node_multi_gpu': multi_node_multi_gpu_launcher
}

try:
    import facebook
    facebook.register_command_launchers(REGISTRY)
except ImportError:
    pass