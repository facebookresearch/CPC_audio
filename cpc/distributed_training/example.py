# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
#####################################################################
# You will find here an example of how to run the distributed mode
# on the FAIR cluster. This is extremly useful for big datasets
#####################################################################

import submitit
import os
from pathlib import Path

#####################################################################

SLURM_LOGS_DIR = Path.home() / "checkpoint" / "20191016_big"
CHECKPOINT_DIR = Path().resolve().parent / "LIBRIBIG"
PATH_DB = "/checkpoint/pem/morgane/LibriBig/"

#####################################################################

os.makedirs(SLURM_LOGS_DIR, exist_ok=True)
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

args = ['--dropout',
        '--hiddenEncoder', '256',
        '--pathCheckpoint', str(CHECKPOINT_DIR),
        '--rnnMode', "transformer",
        '--samplingType', "samespeaker",
        '--save_step', '1',
        '--distributed',
        '--nGPU', '1',
        '--ignore_cache',
        '--batchSizeGPU', '32',
        '--master_port', '18362',
        '--file_extension', ".wav",
        '--restart',
        '--pathDB', PATH_DB]

# submission interface (logs are dumped in the folder)
executor = submitit.AutoExecutor(folder=str(SLURM_LOGS_DIR))
executor.update_parameters(timeout_min=60 * 24 * 3, mem_gb=128,
                           gpus_per_node=8, tasks_per_node=8, nodes=1,
                           partition="dev,priority,learnfair",
                           comment='ICASSP: LibriBIG dataset', name='LIBRIBIG')


def main(args):
    import sys
    sys.path.append('..')
    import train
    return train.main(args)


job = executor.submit(main, args)
print(f"Slurm job submitted. ID: {job.job_id}")
