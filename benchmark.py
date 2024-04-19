import os
if os.path.isdir('/scratch/dmpowell'):
    os.environ['TRANSFORMERS_CACHE'] = '/scratch/dmpowell/.cache/huggingface'
    os.environ['HF_DATASETS_CACHE'] = '/scratch/dmpowell/.cache/huggingface/datasets'
print(os.getenv('TRANSFORMERS_CACHE'))
print(os.getenv('HF_DATASETS_CACHE'))

import numpy as np
import torch
from transformers import GPTJForCausalLM, AutoTokenizer, AutoModel, GPT2LMHeadModel, AutoModelForCausalLM

import pandas as pd
import json
import janitor

from easyeditor.util import nethook
from easyeditor.custom import * # gets my custom functions

import torch.nn.functional as F

from contextlib import redirect_stdout

device = torch.device("cuda")


## --- set up test mode (or not)
MODE_ARGS = [] # []

## --- load data

def load_result(filename):
    x = pd.read_csv(filename, converters={'fwd_choices':literal_eval, 'rev_choices':literal_eval})
    return(x)

baseline_df, edits_df, eval_df = load_data()

# prefix_fwd, prefix_rev, prefix_single = load_prefixes(verbose = False)


## -- set up models and do edits with different methods

hparam_config = dict()
results = dict()

hparam_config["ROME"] = {"HyperParams": ROMEHyperParams, "path": 'hparams/ROME/llama-7b.yaml', "edit_method": "ROME"}
hparam_config["ICE"] = {"HyperParams": ROMEHyperParams, "path": 'hparams/ROME/llama-7b.yaml', "edit_method": "ICE"}
hparam_config["FT"] = {"HyperParams": FTHyperParams, "path": 'hparams/FT/llama-7b.yaml', "edit_method": "FT"}

if "testing" not in MODE_ARGS:
    if "read_baseline" not in MODE_ARGS:
        print("\n\n... estimating baseline performance ...\n\n")
        hparams = FTHyperParams.from_hparams('hparams/FT/llama-7b.yaml')
        edited_model = EditedModel(hparams, auth_token())
        results_baseline = evaluate(baseline_df, edited_model, prefix_fwd = "", prefix_rev = "", normalization = None)
        results_baseline.to_csv("results/csv/" + hparams.model_name.replace("/", "-") + "-baseline" +  ".csv", index=False)
    else:
        print("\n\n... reading in baseline performance ...\n\n")
        results_baseline = load_result("results/csv/meta-llama-Llama-2-7b-hf-baseline.csv")



if "sampling" in MODE_ARGS:
    import random
    random.seed(456)
    edits_df = edits_df.sample(100)

print("\n\n ... editing and evaluating for ...")
print(len(edits_df), " edits\n\n")

for edit_method, HPARAMS in hparam_config.items():    

    hparams = HPARAMS["HyperParams"].from_hparams(HPARAMS["path"])
    
    if "testing" in MODE_ARGS:
        print("dataset testing mode ...")
        res = test_dataset(edits_df, eval_df, None, edit_method = "ICE")
        print(res.agg(fwd=('corr_fwd_answers', 'mean'), rev=('corr_rev_answers', 'mean')))
    
    else:
        print("... estimating edit performance for: ", edit_method, "\n\n" )
        
        edited_model = EditedModel(hparams, auth_token())

        res = edit_and_evaluate(
            edits_df, 
            eval_df, 
            edited_model, 
            edit_method, 
            prefix_fwd = "", 
            prefix_rev = "", 
            log_file = "results/catprop-2024-03-25.txt"
            )
    
        res.to_csv("results/csv/" + hparams.model_name.replace("/", "-") + "-" + edit_method +  "-catprop.csv", index=False)

        results[HPARAMS["edit_method"]] = res
    
    # free up memory to load the next model
    del edited_model
    torch.cuda.empty_cache()