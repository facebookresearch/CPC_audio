# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import json
import numpy as np
from copy import deepcopy
def untensor(d):
    if isinstance(d, list):
        return [untensor(v) for v in d]
    if isinstance(d, dict):
        return dict((k, untensor(v)) for k, v in d.items())
    if hasattr(d, 'tolist'):
        return d.tolist()
    return d


def save_logs(data, pathLogs):
    with open(pathLogs, 'w') as file:
        json.dump(data, file, indent=2)

def update_logs(logs, logStep, prevlogs=None):
    out = {}
    for key in logs:
        out[key] = deepcopy(logs[key])

        if prevlogs is not None:
            out[key] -= prevlogs[key]
        out[key] /= logStep
    return out


def show_logs(text, logs):
    print("")
    print('-'*50)
    print(text)

    for key in logs:

        if key == "iter":
            continue

        nPredicts = logs[key].shape[0]

        strSteps = ['Step'] + [str(s) for s in range(1, nPredicts + 1)]
        formatCommand = ' '.join(['{:>16}' for x in range(nPredicts + 1)])
        print(formatCommand.format(*strSteps))

        strLog = [key] + ["{:10.6f}".format(s) for s in logs[key]]
        print(formatCommand.format(*strLog))

    print('-'*50)
