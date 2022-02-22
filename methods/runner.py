import importlib
import json
import sys

import torch


def run_main(json_ctx=None):
    if not json_ctx and len(sys.argv) <= 1:
        print("\nNo config file is specified.\n")
        return
    elif json_ctx:
        ctx = json_ctx
    else:
        try:
            with open(sys.argv[1], "r") as fhandle:
                ctx = json.load(fhandle)
        except FileNotFoundError:
            print("\nFile {} not found !\n".format(sys.argv[1]))
            return

    command = ctx["command"]
    method = ctx["method"]
    torch.cuda.set_device(int(ctx["gpu"]))

    if command == 'train':
        selected_method = importlib.import_module('methods.{}'.format(method))
        selected_method.cmd_train(ctx)


if __name__ == '__main__':
    run_main()
