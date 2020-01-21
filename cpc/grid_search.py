# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import pathlib
import argparse
import time

from cpc.utils.grid_utils import SlurmWrapper, sweep


if __name__ == '__main__':
    import submitit
    import train

    parser = argparse.ArgumentParser(
        description="A stool-like slurm-compatible grid search tool")
    parser.add_argument("--sweep", action='append', help="Json files with sweep params in the stool format."
                                                         "It is possible to specify several files: --sweep file1.json --sweep file2.json")

    parser.add_argument("--dry_run", action="store_true",
                        help="Synonym for preview")
    parser.add_argument("--name", type=str, default=None,
                        help="sbatch name of job. Also used as the output directory")
    parser.add_argument("--constraint", type=str, default='',
                        help="slurm constraint on the nodes")
    parser.add_argument("--ncpu", type=int, default=8,
                        help="sbatch number of cpus required per task")
    parser.add_argument("--ngpu", type=int, default=1,
                        help="Number of gpus required per task (--gres=gpu:N in sbatch)")
    parser.add_argument("--partition", type=str,
                        default="dev", help="Partition requested")
    parser.add_argument("--time", type=int, default=4320, help="Job timeout")
    parser.add_argument("--comment", type=str, help="slurm comment")

    args = parser.parse_args()

    if not args.name:
        raise ValueError("You have to specify the experiment name!")

    root_dir = (pathlib.PosixPath('~/cpc') / args.name /
                time.strftime("%Y_%m_%d_%H_%M_%S")).expanduser()
    root_dir.mkdir(parents=True)

    executor = submitit.AutoExecutor(folder=root_dir.expanduser())

    executor.update_parameters(timeout_min=args.time, partition=args.partition,
                               cpus_per_task=args.ncpu, gpus_per_node=args.ngpu, name=args.name,
                               comment=args.comment, constraint=args.constraint)

    jobs = []

    comb_id = 0
    for sweep_file in args.sweep:
        for comb in sweep(sweep_file):
            checkpoint_path = root_dir / f'{comb_id}'
            checkpoint_path.mkdir()

            runner = SlurmWrapper(train.main)

            checkpoint_arg = f'--pathCheckpoint={checkpoint_path}'
            comb.append(checkpoint_arg)

            if not args.dry_run:
                job = executor.submit(runner, comb)
                print(f'job id {job.job_id}, args {comb}')
                jobs.append(job)
            else:
                print(f'{comb}')
            comb_id += 1

    print(f'Total jobs launched: {len(jobs)}')
