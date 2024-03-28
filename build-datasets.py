import pandas as pd
import random
import numpy as np

random.seed(123)

def load_template_data(supercat):
    types_df = pd.read_csv("taxonomy/" + supercat + "-type-tokens.tsv", sep="\t")
    properties_df = pd.read_csv("taxonomy/" + supercat +"-data.tsv", sep="\t")

    return(types_df, properties_df)

    
def make_longtypes(types_df):
    longtypes_df = (
        types_df
        .melt(["entity_type"], var_name = "token_type", value_name = "subj")
        .assign(token_type = lambda x: np.where(x.token_type.str.startswith("typical"), "typical", "rare"))
    )

    return(longtypes_df)


def proc_rev_choices(row):
    tokens = (
        properties_df
        .groupby("entity")
        .answer_fwd.agg(anymatches = lambda x: np.any(x == row.answer_fwd))
        .reset_index()
        .loc[lambda x: ~x.anymatches]
        .filter(["entity"])
        # .loc[lambda x: all(x.answer_fwd != row.answer_fwd)]
        # .drop_duplicates(subset = ["entity"])
        .merge(make_longtypes(types_df), how = "left", left_on = "entity", right_on = "entity_type")
        .loc[lambda x: x.token_type == row.token_type]
        .drop_duplicates(subset = ["subj"])
        .subj.to_list()
    )

    
    # k = min(3, len(tokens)) # ... but there aren't enough in some cases
    try: 
        foils = [e for e in list(set(tokens)) if e != row.subj]
        # foils = random.sample([e for e in list(set(tokens)) if e != row.subj], k = 3) # sample WITHOUT replacement
    except:
        return np.nan

    return [row.subj] + foils



def proc_fwd_choices(df, baseline = False):

    if baseline:
        fwd_choice_list = df[["foil1", "foil2", "foil3"]].values.tolist()
        
    else:
        fwd_choice_list = df[["foil1", "foil2", "foil3", "orig_answer_fwd"]].values.tolist()
    ans_list = df["answer_fwd"].tolist()
    out = []

    for i in range(len(fwd_choice_list)):
        distinct = list(set(fwd_choice_list[i]))
        ans = ans_list[i]
        out.append([ans] + [c for c in distinct if c!=ans and pd.notna(c)])

    df["fwd_choices"] = out

    
    return(df)


def proc_rev_choices_catprop(row):

# if subj is in entity list, then use entities
# if subj is not, then use another token example of the same type

    rel_entities = (
        properties_df
        .loc[lambda x: x.property == row.property]
    ).entity.tolist()

    if len(rel_entities) == 1:
        rel_entities = properties_df.entity.tolist()

    entities = (
        properties_df
        .loc[lambda x: x.entity.isin(rel_entities)]
        .groupby("entity")
        .answer_fwd.agg(anymatches = lambda x: np.any(x == row.answer_fwd))
        .reset_index()
        .loc[lambda x: ~x.anymatches]
        .filter(["entity"])
        # .loc[lambda x: all(x.answer_fwd != row.answer_fwd)]
        .drop_duplicates(subset = ["entity"])
        # .merge(make_longtypes(types_df), how = "left", left_on = "entity", right_on = "entity_type")
        # .loc[lambda x: x.token_type == row.token_type]
        # .drop_duplicates(subset = ["subj"])
        .entity.to_list()
    )

    if row.subj not in properties_df.entity.tolist():
        
        tokens = (
            make_longtypes(types_df)
            .loc[lambda x: x.entity_type.isin(entities)]
            .loc[lambda x: x.token_type == row.token_type]
            .drop_duplicates(subset = ["subj"])
        ).subj.tolist()

    else:
        tokens = entities


    # k = min(3, len(tokens)) # ... but there aren't enough in some cases
    try: 
        foils = [e for e in list(set(tokens)) if e != row.entity]
        # foils = random.sample([e for e in list(set(tokens)) if e != row.subj], k = 3) # sample WITHOUT replacement
    except:
        return np.nan
    
    return [row.subj] + foils
    

def filter_choice_lists(df):

    out = (
        df
        .assign(
            fwd_choice_len = lambda d: d.apply(lambda x: len(x.fwd_choices), 1),
            rev_choice_len = lambda d: d.apply(lambda x: len(x.rev_choices), 1)
        )
        .loc[lambda x: x.rev_choice_len >= 2]
        .loc[lambda x: x.fwd_choice_len >= 2]
        .drop(["fwd_choice_len", "rev_choice_len"], axis = 1)
    )

    return(out)


def make_catmem_edits_df(types_df, properties_df):

    longtypes_df = make_longtypes(types_df)

    edits_df = (
        pd.merge(types_df, types_df, how = "cross")
        .loc[lambda x: x.entity_type_x!=x.entity_type_y] 
        .filter(['entity_type_x', 'entity_type_y', 'typical_token1_y', 'rare_token1_y', 'typical_token2_y', 'rare_token2_y'])
        .rename(columns = {"entity_type_y": "orig_entity"})
        .melt(['entity_type_x', "orig_entity"])    
        .rename(columns={"entity_type_x":"entity", "value":"subj", "variable":"token_type"})
        .assign(token_type = lambda x: np.where(x.token_type.str.startswith("typical"), "typical", "rare"))
        .assign(edit = lambda x: x.subj + " -> " + x.entity)
        .assign(edit_type = "category membership")
    )
    return(edits_df)


def make_edits_df(type_df, properties_df):
    dfs = [make_catmem_edits_df(type_df, properties_df)]
    return(pd.concat(dfs))


def make_baseline_df(types_df, properties_df):
    longtypes_df = make_longtypes(types_df)

    baseline_df = (
        types_df
        .rename(columns = {'entity_type':'entity'})
        .melt(["entity"], value_name = 'subj')
        .merge(properties_df, on = 'entity')
        .assign(orig_entity = lambda x: x.entity)
        .pipe(proc_fwd_choices, True)
        .rename(columns = {"variable":"token_type"})
        .assign(token_type = lambda x: np.where(x.token_type.str.startswith("typical"), "typical", "rare"))
        .assign(rev_choices = lambda x: x.apply(proc_rev_choices, 1))
    )

    baseline_cat_members = (
        longtypes_df
        .assign(
            category_membership = "a <subj> is a kind of <answer>",
            # category_membership1 = "which is where the name originates. In any case, all <subj> are <answer>",
            # category_membership2 = "it is correct to say that any <subj> is a <answer>",
            # category_membership3 = "a <subj> is one variety of <answer>"
            )
        .melt(id_vars = ["entity_type", "token_type", "subj"],  var_name = "property", value_name = "query_fwd")
        .assign(
            query_rev = longtypes_df
                        .assign(
                            category_membership = "a <subj> is a kind of <answer>",
                            # category_membership1 = "which is where the name originates. In any case, all <subj> are <answer>",
                            # category_membership2 = "it is correct to say that any <subj> is a <answer>",
                            # category_membership3 = "a <subj> is one variety of <answer>"
                            )
                        .melt(id_vars = ["entity_type", "token_type", "subj"], var_name = "property", value_name = "query_rev")
                        .query_rev
        )
        .assign(
            fwd_choices = lambda d: d.apply(lambda x: [x.entity_type] + [t for t in types_df.entity_type.to_list() if t != x.entity_type], 1),
            rev_choices = lambda d: d.apply(lambda x: [x.subj] + list(set([t for t in d.loc[(d.entity_type != x.entity_type) & (d.token_type == x.token_type)].subj.to_list()])), 1),
            answer_fwd = lambda x: x.entity_type,
            answer_rev = lambda x: x.subj
        )
        .rename(columns = {'entity_type':'entity'})
    )


    # baseline_category_property_df = (
    #     baseline_df
    #     .assign(subj = lambda x: x.entity)
    #     .assign(rev_choices = lambda d: d.apply(lambda x: longtypes_df.loc[longtypes_df.subj.isin(x.rev_choices)].entity_type.to_list(), 1)) #longtypes_df.loc[longtypes_df.subj.isin(x.rev_choices)].entity_type)
    #     .drop_duplicates(subset = ["entity", "property"])
    #     .assign(token_type = "entity")
    #     .assign(
    #         # answer_fwd = lambda x: x.entity,
    #         answer_rev = lambda x: x.subj
    #     )
    #     .rename(columns = {'entity_type':'entity'})
    # )

    baseline_category_property_df = (
        baseline_df
        .assign(subj = lambda x: x.entity)
        .assign(rev_choices = lambda x: x.apply(proc_rev_choices_catprop, 1))
        .drop_duplicates(subset = ["entity", "property"])
        .assign(token_type = "entity")
        .assign(
            answer_rev = lambda x: x.subj
        )
        .rename(columns = {'entity_type':'entity'})
    )

    return(pd.concat([baseline_cat_members, baseline_category_property_df, baseline_df]))






def make_catmem_eval_df(type_df, properties_df):

    edits_df = make_catmem_edits_df(type_df, properties_df)
    longtypes_df = make_longtypes(types_df)
    
    eval_df = ( 
        pd.merge(
            edits_df, 
            properties_df.filter(["entity", "answer_fwd", "answer_rev", "property"]).rename(columns = {"answer_fwd":"orig_answer_fwd", "answer_rev":"orig_answer_rev", "entity":"orig_entity"}), 
            how="left", on = "orig_entity"
            )
            .merge(properties_df, on = ["entity", "property"]) 
            # .loc[lambda x: x.orig_answer_fwd!=x.answer_fwd]
            .assign(answer_changed = lambda x: x.orig_answer_fwd!=x.answer_fwd )
            .pipe(proc_fwd_choices)
            .rename(columns = {"variable":"token_type"})
            .assign(rev_choices = lambda x: x.apply(proc_rev_choices, 1))
    )

    eval_cat_members = (
        edits_df
        .assign(
                            category_membership = "a <subj> is a kind of <answer>",
                            # category_membership1 = "which is where the name originates. In any case, all <subj> are <answer>",
                            # category_membership2 = "it is correct to say that any <subj> is a <answer>",
                            # category_membership3 = "a <subj> is one variety of <answer>"
                            )
        .melt(id_vars = ["entity", "orig_entity", "token_type", "edit_type", "edit", "subj"],  var_name = "property", value_name = "query_fwd")
        .assign(
            query_rev = edits_df
                        .assign(
                            category_membership = "a <subj> is a kind of <answer>",
                            # category_membership1 = "which is where the name originates. In any case, all <subj> are <answer>",
                            # category_membership2 = "it is correct to say that any <subj> is a <answer>",
                            # category_membership3 = "a <subj> is one variety of <answer>"
                            )
                        .melt(id_vars = ["entity", "orig_entity", "token_type", "edit_type", "edit", "subj"],  var_name = "property", value_name = "query_rev")
                        .query_rev
        )
        .assign(answer_changed = True)
        .assign(
                fwd_choices = lambda d: d.apply(lambda x: [x.entity] + [t for t in types_df.entity_type.to_list() if t != x.entity], 1),
                
                rev_choices = lambda d: d.apply(lambda x: [x.subj] + list(set([t for t in longtypes_df.loc[lambda t: (t.entity_type != x.orig_entity) & (t.token_type == x.token_type) & (t.entity_type != x.entity)].subj.to_list()])), 1),
                answer_fwd = lambda x: x.entity,
                answer_rev = lambda x: '<subj>',
                orig_answer_fwd = lambda x: x.orig_entity
            )
    )

    out = pd.concat([eval_cat_members, eval_df]).pipe(filter_choice_lists)

    return(out)




def make_eval_df(type_df, properties_df):
    dfs = [make_catmem_eval_df(type_df, properties_df)]
    return(pd.concat(dfs))


baseline_df = pd.DataFrame()
edits_df = pd.DataFrame()
eval_df = pd.DataFrame()

for supercat in ["animal","plant", "vehicle", "instrument", "food", "drink"]:#

    types_df, properties_df = load_template_data(supercat)

    baseline_df = pd.concat([baseline_df, make_baseline_df(types_df, properties_df).assign(superordinate_category = supercat) ])
    
    edits = make_edits_df(types_df, properties_df).assign(superordinate_category = supercat)
    edits_df = pd.concat([edits_df, edits])
    
    evals = make_eval_df(types_df, properties_df).assign(superordinate_category = supercat)
    eval_df = pd.concat([eval_df, evals])

print(len(edits_df), " edits")
print(len(eval_df)," evaluations")
print("--------- writing to json ...")
edits_df.reset_index().to_json("edits.json")
baseline_df.reset_index().to_json("baseline-evaluation.json")
eval_df.reset_index().to_json("edits-evaluation.json")