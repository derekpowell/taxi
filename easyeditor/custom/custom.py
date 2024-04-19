from ..editors import BaseEditor
from ..models.rome import ROMEHyperParams
from ..models.ft import FTHyperParams
from ..models.pmet import PMETHyperParams
from ..models.grace import GraceHyperParams
from ..models.memit import MEMITHyperParams

from ..util import nethook

import transformers
import torch
import torch.nn.functional as F
from transformers import GPTJForCausalLM, AutoTokenizer, AutoModel, GPT2LMHeadModel, AutoModelForCausalLM
import pandas as pd

from contextlib import redirect_stdout
import sys

from tqdm import tqdm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def pad_token(token):
    # token = " " + token if token[0] != " " else token
    token = " " + token.lstrip()
    return(token)


def encode_token(token:str, tokenizer, pad = True):        
    
    # deal with sentencepiece tokenizer
    if type(tokenizer) == transformers.models.llama.tokenization_llama.LlamaTokenizer:
        # token = pad_token(token) if pad else token # another edit 2024-02-05
        token_id = tokenizer(token)["input_ids"]
        return token_id[1:]
    else:
        token = pad_token(token) if pad else token
        token_id = tokenizer(token)["input_ids"]

        return(token_id)


class EditedModel:
    def __init__(self, hparams, auth_token=None):
        self.editor = BaseEditor.from_hparams(hparams)

        self.model = self.editor.model
        self.tok = self.editor.tok
        self.model_name = self.editor.model_name
        

        self.params = hparams
        self.preprompt = ""
        self.saved_weights = None
        
        self.tok.padding_side = "left"
        # self.tok.pad_token = self.tok.eos_token
    
    def edit(self, rewrite, log_file = None, **kwargs):
        if log_file:
            h = open(log_file, "a")
        else:
            h = None
        
        if "preprompt" in rewrite: # this is a little hacky
            self.preprompt = rewrite["preprompt"]
            return None
        
        # elif type(rewrite) == dict:
        else:
            with redirect_stdout(h): # None
                metrics, self.model, self.saved_weights = self.editor.pure_edit( # pure_edit
                    **rewrite,
                    # **kwargs,
                    keep_original_weight = True,
                    verbose = False
                )
        # elif type(rewrite)==list:

        #     # prompts = [x['prompts'] for x in rewrite]
        #     # target_new = [x['target_new'] for x in rewrite]

        #     with redirect_stdout(h): # None
        #         metrics, self.model, self.saved_weights = self.editor.pure_edit( # pure_edit
        #             rewrite,
        #             # target_new,
        #             # **kwargs,
        #             keep_original_weight = True,
        #             verbose = False
        #         )
        

        return metrics
    
    
    def restore(self):

        self.preprompt = ""
        
        if self.saved_weights == {}:
            print (print(f"No model weights to restore: saved_weights is empty dict"))

        elif self.saved_weights:

            try:
                with torch.no_grad():
                    for k, v in self.saved_weights.items():
                        nethook.get_parameter(self.model, k)[...] = v
                self.saved_weights = None
                # print("Original model restored")
            except NameError as e:
                print(f"No model weights to restore: {e}")

            
    def generate_text(self, texts, **kwargs):
        
        if type(texts) != list:
            texts = [texts]
        
        texts = [self.preprompt + t for t in texts]

        model = self.model
        tokenizer = self.tok
        encoding = tokenizer(texts, padding=True, return_tensors='pt').to(device)

        with torch.no_grad():
            generated_ids = model.generate(**encoding, **kwargs) # 

            generated_texts = tokenizer.batch_decode(
                generated_ids, skip_special_tokens=True
            )
            
        return(generated_texts)
    
    
    def logprobs(self, texts):
        
        texts = self.preprompt + texts if type(texts)==str else [self.preprompt + t for t in texts]
    
        tokenizer = self.tok 
        model = self.model
        encoding = tokenizer(texts, padding=True, return_tensors='pt').to(device)

        with torch.no_grad():
            model_out = model(encoding["input_ids"])
            logits = model_out.logits
            logprobs = F.log_softmax(logits, -1)
        
        return {"tokens": encoding, "logprobs": logprobs}

    
    def completion_logprob(self, text, completion, start_ind = None):
        
        '''
        Compute model log probability of completion substring. Returns single value tensor. Takes only one text string.
        '''
        
        # texts = self.preprompt + text
    
        # tokenizer = self.tok 
        # model = self.model
        # encoding = tokenizer(texts, padding=True, return_tensors='pt').to(device)

        # with torch.no_grad():
        #     model_out = model(encoding["input_ids"])
        #     logits = model_out.logits
        #     logprobs = F.log_softmax(logits, -1)

        # token_id = encode_token(completion, tokenizer)
        # start_ind = -len(token_id)-1 if not start_ind else start_ind
        
        # l = logprobs[:, start_ind:-1, token_id]
        # if len(l.squeeze().shape) == 0:
        #     return(l.squeeze())
        # else:
        #     return(l.squeeze().diag().sum())
        

        return self.substring_logprobs(text, completion)[0][-1]
        

    def substring_logprobs(self, texts, substring, pad = True):
        '''
        Compute model log probability of each occurrence of substring in text. Returns list of list-type. Accepts a list of strings.
        '''
        
        if type(texts) != list:
            texts = [texts]
        
        logprobs = self.logprobs(texts)
        
        tok_encoded = encode_token(substring, self.tok, pad = pad)
        # text_encoded = logprobs['tokens']['input_ids'][0].tolist()
        
        out = []
        for i in range(len(texts)):
            text_encoded = logprobs['tokens']['input_ids'][i].tolist()

            # find matches for searched token sequence
            start_idxs = []
            for left in range(0, len(text_encoded) - len(tok_encoded)+1):
                # left = i - 1
                right = left + len(tok_encoded)
                if text_encoded[left:right] == tok_encoded:
                    start_idxs.append(left)

            lp = logprobs['logprobs'][i]
            match_probs = []

            # compute probability for all tokens
            for start in start_idxs:
                val = 0
                for i in range(len(tok_encoded)):
                    val += lp[start + i - 1][tok_encoded[i]]
                match_probs.append(val)

            out.append(match_probs)

        return out
        

    def choose(self, prompt, choices, normalization = None):

        # prompt = prompt.rstrip() # remove any trailing whitespace

        if type(self.tok) == transformers.models.llama.tokenization_llama.LlamaTokenizer:
            padded_choices = choices
            prompt = prompt + " " if prompt[-1]!= " " else prompt
        else:
            padded_choices = [pad_token(c) for c in choices] # pad all the 
        
        prompts = [prompt + c for c in padded_choices]

        logits = torch.tensor([self.completion_logprob(prompts[i], padded_choices[i]) for i in range(len(padded_choices))])

        if normalization == "unconditional":
            norm_logits = torch.tensor([self.completion_logprob(padded_choices[i], padded_choices[i]) for i in range(len(padded_choices))])
            logits = logits - norm_logits

        elif normalization == "byte_length":    
            str_lens = [len(c) for c in choices]
            logits = logits / torch.tensor(str_lens)

        elif normalization == "token_length":
            tok_lens = [len(encode_token(c, self.tok)) for c in choices]
            logits = logits / torch.tensor(tok_lens)

        elif normalization == "root":
            tok_lens = [len(encode_token(c, self.tok)) for c in choices]
            logits = torch.pow(torch.exp(logits), 1./torch.tensor(tok_lens))

        logits = logits.tolist()

        return(logits.index(max(logits)))
    

def evaluate(evaluation_data, model, prefix_fwd = "", prefix_rev = "", normalization = None):

    fwd_answers = []
    rev_answers = []
    corr_fwd_answers = []
    corr_rev_answers = []

    for q in evaluation_data.itertuples():

        fwd_choices =  q.fwd_choices
        query_fwd = q.query_fwd.replace("<subj>", q.subj).replace("<answer>", "")
        if q.property not in ["category_membership", "category_membership1", "category_membership2","category_membership3"]: # do not use prefix for these
            query_fwd = prefix_fwd + query_fwd
        ans_fwd = model.choose(query_fwd, fwd_choices, normalization = normalization) # None, "unconditional", "byte_length", "token_length", "root"
        corr_fwd_answers.append(fwd_choices.index(q.answer_fwd))
        fwd_answers.append(ans_fwd)

        rev_choices =  q.rev_choices
        query_rev = q.query_rev.replace("<answer>", q.answer_fwd).replace("<subj>", "")
        if q.property not in ["category_membership", "category_membership1", "category_membership2","category_membership3"]: # do not use prefix for these
            query_rev = prefix_rev + query_rev
        ans_rev = model.choose(query_rev, rev_choices, normalization = normalization) # None, "unconditional", "byte_length", "token_length", "root"
        corr_rev_answers.append(rev_choices.index(q.subj))
        rev_answers.append(ans_rev)

    results = (
        evaluation_data
        .assign(
            corr_fwd_answer = corr_fwd_answers,
            corr_rev_answer = corr_rev_answers,
            fwd_predicted = fwd_answers,
            rev_predicted = rev_answers
            )
        .assign(
            correct_fwd = lambda x: x.corr_fwd_answer==x.fwd_predicted,
            correct_rev = lambda x: x.corr_rev_answer==x.rev_predicted
        )
    )

    return(results)


def make_edit_batches(df):
    df2 = df.copy()
    batches = []
    while df2.shape[0] > 0:
        batch = df2.groupby(["entity"]).sample(1)
        batches.append(batch)
        df2 = df2.loc[lambda x: ~x.edit.isin(batch.edit)]

    return(batches)


def make_rewrite(e):
    rewrite = {
            'prompt': f'A {e.subj} is a kind of',
            'target_new': e.entity, #{'str': e.entity},
            'subject': e.subj
            }
    
    return(rewrite)


def edit_and_evaluate(edits_df, eval_df, model, edit_method, metrics = False, log_file = None, **kwargs):
    
    full_results = pd.DataFrame()
    result_list = []
    full_metrics = []
    print("===== Editing and evaluating =====")

    if edit_method in ["MEMIT", "PMET"]:
        print("making batches ...")

        batches = make_edit_batches(edits_df)
        print("editing in batches ...")
        for b in tqdm(batches):
            
            rewrites = b.apply(make_rewrite, 1).to_list()

            rewrites = {"prompts": [x["prompt"] for x in rewrites], "target_new": [x["target_new"] for x in rewrites], "subject": [x["subject"] for x in rewrites] }
            
            metrics = model.edit(rewrites, log_file  = log_file)

            evals = b.filter(["edit"]).merge(eval_df, how = "left", on = "edit")
            
            res = evaluate(evals, model, **kwargs)
            
            model.restore()

            full_results = pd.concat([full_results, res])
    else:

        for e in tqdm(edits_df.itertuples(),  total = edits_df.shape[0]):
            if e.edit_type == "category membership":
                if edit_method in ["ROME", "FT", "GRACE"]:
                    rewrite = {
                            'prompts': [f'A {e.subj} is a kind of'],
                            'target_new': [e.entity], #{'str': e.entity},
                            'subject': [e.subj]
                            }
                    metrics = model.edit(rewrite, log_file  = log_file)
                    full_metrics.append(metrics)
                elif edit_method == "ICE":
                    model.edit({"preprompt": f"Imagine that a {e.subj} is a kind of {e.entity} ...\n\n"}) # and not a kind of {e.orig_entity}
                    
                elif edit_method == "BASE":
                    model.edit({"preprompt": ""})


                evals = eval_df.loc[lambda x: (x.edit_type == "category membership") & (x.entity == e.entity) & (x.subj == e.subj)]

            elif e.edit_type == "category property":
                if edit_method in ["ROME", "FT", "PMET", "GRACE"]:
                    rewrite_prompt = e.query_fwd.replace("<subj>", e.entity).replace(" <answer>", "")
                    rewrite = {
                        'prompts': [rewrite_prompt],
                        'target_new': [e.answer_fwd], #{'str': e.entity},
                        'subject': [e.entity]
                    }
                    metrics = model.edit(rewrite, log_file  = log_file)
                    full_metrics.append(metrics)

                elif edit_method == "ICE":
                    
                    rewrite_prompt = e.query_fwd.replace("<subj>", e.entity).replace("<answer>", e.answer_fwd)
                    model.edit({"preprompt": f"Imagine that {rewrite_prompt} ...\n\n"}) # and not a kind of {e.orig_entity}    

                evals = eval_df.loc[lambda x: (x.edit == e.edit)]
            
            result = evaluate(evals, model, **kwargs)
            result_list.append(result)
        
        model.restore()

    
    full_results = pd.concat(result_list)    

    full_results["edit_method"] = edit_method
    
    return(full_results)



def test_dataset(edits_df, eval_df, model, edit_method=None, prefix_fwd = "", prefix_rev = "", metrics = False, log_file = None, **kwargs):
    # just runs through the data without actually using the model
    full_results = pd.DataFrame()
    full_metrics = []

    for e in edits_df.itertuples():
        if e.edit_type == "category membership":
            if edit_method in ["ROME", "FT", "PMET", "GRACE"]:
                rewrite = {
                        'prompts': [f'A {e.subj} is a kind of'],
                        'target_new': [e.entity], #{'str': e.entity},
                        'subject': [e.subj]
                        }
                # metrics = model.edit(rewrite, log_file  = log_file)
                # full_metrics.append(metrics)
            elif edit_method == "ICE":
                rewrite_prompt = {"preprompt": f"Imagine that a {e.subj} is a kind of {e.entity} ...\n\n"}
            
            evals = eval_df.loc[lambda x: (x.edit_type == "category membership") & (x.entity == e.entity) & (x.subj == e.subj)]

        elif e.edit_type == "category property":
            if edit_method in ["ROME", "FT", "PMET", "GRACE"]:
                rewrite_prompt = e.query_fwd.replace("<subj>", e.entity).replace("<answer>", e.answer_fwd)
               
                rewrite = {
                    'prompts': [rewrite_prompt],
                    'target_new': [e.answer_fwd], #{'str': e.entity},
                    'subject': [e.entity]
                }

                print(rewrite)
                # metrics = model.edit(rewrite, log_file  = log_file)
                # full_metrics.append(metrics)

            elif edit_method == "ICE":
                
                rewrite_prompt = e.query_fwd.replace("<subj>", e.entity).replace("<answer>", e.answer_fwd)
                print(f"Imagine that {rewrite_prompt} ...\n\n")
                # model.edit({"preprompt": f"Imagine that {rewrite_prompt} ...\n\n"}) # and not a kind of {e.orig_entity}    

            evals = eval_df.loc[lambda x: (x.edit_type == "category property") & (x.entity == e.entity) & (x.property == e.property)]
        
        res = test_eval_data(evals, model, prefix_fwd, prefix_rev, **kwargs)
        
        # model.restore()

        full_results = pd.concat([full_results, res])

    full_results["edit_method"] = edit_method
    
    return(full_results)



def test_eval_data(evaluation_data, model, prefix_fwd = "", prefix_rev = ""):
    # just runs through the data without actually using the model

    fwd_answers = []
    rev_answers = []
    corr_fwd_answers = []
    corr_rev_answers = []

    for q in evaluation_data.itertuples():

        fwd_choices =  q.fwd_choices
        query_fwd = q.query_fwd.replace("<subj>", q.subj).replace("<answer>", "")
        if q.property not in ["category_membership", "category_membership1", "category_membership2","category_membership3"]: # do not use prefix for these
            query_fwd = prefix_fwd + query_fwd
        # ans_fwd = model.choose(query_fwd, fwd_choices, normalization = None) # None, "unconditional", "byte_length", "token_length", "root"

        try:
            z = fwd_choices.index(q.answer_fwd)
            corr_fwd_answers.append(1)
        except:
            print("check forward answers!", q.edit, q.property)
            corr_fwd_answers.append(0)

        rev_choices =  q.rev_choices
        query_rev = q.query_rev.replace("<answer>", q.answer_fwd).replace("<subj>", "")

        if q.property not in ["category_membership", "category_membership1", "category_membership2","category_membership3"]: # do not use prefix for these
            query_rev = prefix_rev + query_rev
        
        try:
            z = rev_choices.index(q.subj)
            corr_rev_answers.append(1)
        except:
            print("check reverse answers!", q.edit, q.property)
            corr_rev_answers.append(0)
        
    results = (
        evaluation_data
        .assign(
            corr_fwd_answers = corr_fwd_answers,
            corr_rev_answers = corr_rev_answers
            )
    )

    return(results)


def filter_evals(baseline_result, edits_df, eval_df):
    corr_memberships = baseline_result.loc[lambda x: x.property=="category_membership"].loc[lambda x: x.correct_fwd]
    # join w/ edits/evals on corr_memberships.subj == edits/evals.subj 
    # when editing dogs, only test labradors if it knew labradors were dogs
    
    category_property_evals = eval_df.loc[lambda x: (x.entity.isin(corr_memberships.entity) & (x.edit_type == "category property"))]

    corr_properties = baseline_result.loc[lambda x:(x.correct_fwd) | (x.property.str.startswith("category_membership"))]
    # join w/ edits/evals on corr_properties.subj = eval_df.entity and property=property
    # when editing whether a dog is a cow, only test on properties it knew cows have
    # and only test "unchanged" properties it knew dogs have

    # corr_properties

    category_membership_evals = (
        pd.merge(
            corr_properties.filter(["entity", "property"]),
            eval_df.loc[lambda x: x.edit_type == "category membership"],
            how = "left",
            on = ["entity", "property"]
        )
    )

    out = pd.concat([category_property_evals, category_membership_evals])

    return(out)


def filter_edits_evals(baseline_result, edits_df, eval_df):
    filtered_evals = filter_evals(baseline_result, edits_df, eval_df)
    filtered_edits = edits_df.loc[lambda x: x.edit.isin(filtered_evals.edit)]

    return(filtered_edits, filtered_evals)
