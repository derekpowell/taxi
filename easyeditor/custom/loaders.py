import pandas as pd

from ast import literal_eval

def load_data(data_path = "datasets/"):

    data_dict = dict()

    edits_df = pd.read_json(data_path + "edits.json")
    baseline_df = pd.read_json(data_path + "baseline-evaluation.json")
    eval_df = pd.read_json(data_path + "edits-evaluation.json")

    return baseline_df, edits_df, eval_df


import configparser

def auth_token():

    config = configparser.ConfigParser()
    config.read("config.ini")
    return config["hugging_face"]["token"]


def load_prefixes(prefix_lines = -1, verbose=False):
    with open('prefix_fwd.txt') as f:
        prefix_fwd = "".join(f.readlines()[0:prefix_lines])

    with open('prefix_rev.txt') as f:
        prefix_rev = "".join(f.readlines()[0:prefix_lines])

    with open('single-prefix2.txt') as f:
        prefix_single = "".join(f.readlines()[0:prefix_lines])

    # prefix_fwd = f.read()
    if verbose:
        print(prefix_fwd)
        print("---")
        print(prefix_rev)
        print("---")

    return prefix_fwd, prefix_rev, prefix_single