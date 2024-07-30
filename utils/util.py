import os
import logging
import argparse

def log_args(args: argparse.Namespace):
    args_text = "\n----------parameters----------\n" + "\n".join([f"{k}: {v}" for k, v in vars(args).items()])
    logging.info(args_text)

def mkdir(path: str):
    if os.path.exists(path):
        pass
    else:
        os.mkdir(path)

def mkdirs(list_of_paths: list[str]):
    for path in list_of_paths:
        mkdir(path)

def write_args(args: argparse.Namespace) -> str:
    # return a string of the arguments
    text = ""
    for k, v in sorted(vars(args).items()):
       text += f"{k}: {v}\n"
    return text