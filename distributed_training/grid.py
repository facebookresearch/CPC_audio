#####################################################################
# You will find here an example of how to run the distributed mode
# on the FAIR cluster. This is extremly useful for big datasets
#####################################################################
import pathlib
import submitit
import os, time


import json
import itertools

def parse_json_sweep(config):
    config = { k: v if type(v) is list else [v] for k, v in config.items() }
    perms = list(itertools.product(*config.values()))

    def to_arg(k, v):
        if type(v) in (int, float):
            return f"--{k}={v}"
        elif type(v) is bool:
            return f"--{k}" if v else None
        elif type(v) is str:
            assert '"' not in v, f"Key {k} has string value {v} which contains forbidden quotes."
            return f'--{k}={v}'
        else:
            raise Exception(f"Key {k} has value {v} of unsupported type {type(v)}.")

    commands = []
    for p in perms:
        args = [to_arg(k, p[i]) for i, k in enumerate(config.keys())]
        args = [arg for arg in args if arg] # filter Nones
        commands.append(args)
    return commands


def sweep(fname):
    with open(fname, 'r') as config_file:
        config = json.loads(config_file.read())
    return parse_json_sweep(config)


def parse_args():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--sweep", action='append', default=[], help="Json file with sweep params in the stool format.")
    parser.add_argument("--dry_run", action="store_true", help="Synonym for preview")

    parser.add_argument("--name", type=str, default=None, help="sbatch name of job. Also used in the output directory")
    parser.add_argument("--partition", type=str, default="dev", help="Partition requested")
    parser.add_argument("--root_path", type=str, help="Path for the run's root dir; using ~/cpc_grid if omitted")
    args = parser.parse_args()

    assert args.name
    if not args.root_path:
        args.root_path = pathlib.Path.home() / 'cpc_grid'

    return args

def main(args):
    import sys
    sys.path.append('..')
    import cpc.train as train
    return train.main(args)

class SlurmWrapper:
    """
    We assume that checkpointing is done within trainer, each epoch.
    """
    def __init__(self, runnable):
        self.runnable = runnable
        self.args = None

    def __call__(self, args):
        self.args = args
        print(f'# launching {json.dumps(args)}', flush=True)
        self.runnable(args)

    def checkpoint(self, _something):
        import submitit

        training_callable = SlurmWrapper(self.runnable)
        return submitit.helpers.DelayedSubmission(training_callable, self.args)


if __name__ == '__main__':
    args = parse_args()

    run_dir = pathlib.PosixPath(args.root_path) / args.name / time.strftime("%Y_%m_%d_%H_%M_%S")
    os.makedirs(run_dir, exist_ok=True)

    combinations = []
    for sweep_file in args.sweep:
        combinations.extend(sweep(sweep_file))

    executor = submitit.AutoExecutor(folder=str(run_dir))
    executor.update_parameters(timeout_min=60 * 24 * 3, mem_gb=128,
                                gpus_per_node=8, tasks_per_node=1, nodes=1,
                                partition=args.partition, cpus_per_task=16,
                                comment='',
                                name=args.name)

    for i, combination in enumerate(combinations):
        job_dir = run_dir / str(i)
        os.makedirs(job_dir, exist_ok=True)
        job_checkpoint = job_dir
        combination += ['--pathCheckpoint', str(job_checkpoint)]

        print(f'Launching [ {" ".join(combination)} ]')
        if not args.dry_run:
            runnner = SlurmWrapper(main)
            job = executor.submit(main, combination)
