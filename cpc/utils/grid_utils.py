# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import json
import itertools


class SlurmWrapper:
    """
    We assume that checkpointing is done within trainer, each epoch.
    """

    def __init__(self, runnable, verbose=False):
        self.runnable = runnable
        self.args = None
        self.verbose = verbose

    def __call__(self, args):
        self.args = args
        if self.verbose:
            print(f'# launching {json.dumps(args)}', flush=True)
        self.runnable(args)

    def checkpoint(self, _):
        import submitit

        training_callable = SlurmWrapper(self.runnable)
        return submitit.helpers.DelayedSubmission(training_callable, self.args)


def parse_json_sweep(config):
    config = {k: v if type(v) is list else [v] for k, v in config.items()}
    perms = list(itertools.product(*config.values()))

    def to_arg(k, v):
        if type(v) in (int, float):
            return f"--{k}={v}"
        elif type(v) is bool:
            return f"--{k}" if v else ""
        elif type(v) is str:
            assert '"' not in v, f"Key {k} has string value {v} which contains forbidden quotes."
            return f'--{k}={v}'
        else:
            raise Exception(
                f"Key {k} has value {v} of unsupported type {type(v)}.")

    commands = []
    for p in perms:
        args = [to_arg(k, p[i]) for i, k in enumerate(config.keys())]
        commands.append(args)
    return commands


def sweep(fname):
    with open(fname, 'r') as config_file:
        config = json.loads(config_file.read())
    return parse_json_sweep(config)
