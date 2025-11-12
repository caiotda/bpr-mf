from collections import defaultdict
import numpy as np
import pandas as pd

def generate_bpr_dataset(interactions_dataset, num_negatives=3):
    user2items = defaultdict(set)
    for u, i in zip(interactions_dataset["user"], interactions_dataset["item"]):
        user2items[u].add(i)

    positives = interactions_dataset[["user", "item"]].to_numpy()
    users_pos, items_pos = positives[:, 0], positives[:, 1]

    triples = []
    # Assuming taht the items are zero-indexed and continuous.
    n_items = interactions_dataset["item"].max() + 1
    all_items = np.arange(n_items)

    for u, i_pos in zip(users_pos, items_pos):
        seen = user2items[u]
        candidates = np.setdiff1d(all_items, list(seen), assume_unique=True)
        
        if len(candidates) >= num_negatives:
            sampled_neg = np.random.choice(candidates, size=num_negatives, replace=False)
        else:
            sampled_neg = np.random.choice(candidates, size=num_negatives, replace=True)

        u_vec = np.repeat(u, len(sampled_neg)) 
        i_pos_vec = np.repeat(i_pos, len(sampled_neg)) 

        triples.append(np.stack([u_vec, i_pos_vec, sampled_neg], axis=1))

    triples = np.vstack(triples)
    return pd.DataFrame(triples, columns=["user", "pos_item", "neg_item"])




def generate_bpr_dataset_with_click_data(interactions_dataset, num_negatives=3):
    user2items = defaultdict(set)
    for u, i in zip(interactions_dataset["user"], interactions_dataset["item"]):
        user2items[u].add(i)

    feedback_df = interactions_dataset[["user", "item", "click"]]
    positives = feedback_df[~feedback_df["click"].isna()].to_numpy()
    users_pos = positives[:, 0].astype(int)
    items_pos = positives[:, 1].astype(int)
    click_pos = positives[:, 2].astype(int)

    tuples = []
    # Assuming taht the items are zero-indexed and continuous.
    n_items = interactions_dataset["item"].max() + 1
    all_items = np.arange(n_items)

    for u, i_pos, click_positions in zip(users_pos, items_pos, click_pos):
        seen = user2items[u]
        candidates = np.setdiff1d(all_items, list(seen), assume_unique=True)
        
        if len(candidates) >= num_negatives:
            sampled_neg = np.random.choice(candidates, size=num_negatives, replace=False)
        else:
            sampled_neg = np.random.choice(candidates, size=num_negatives, replace=True)

        u_vec = np.repeat(u, len(sampled_neg)) 
        i_pos_vec = np.repeat(i_pos, len(sampled_neg)) 
        click_pos_vec = np.repeat(click_positions, len(sampled_neg)) 

        tuples.append(np.stack([u_vec, i_pos_vec, click_pos_vec, sampled_neg], axis=1))

    tuples = np.vstack(tuples)
    return pd.DataFrame(tuples, columns=["user", "pos_item", "click_position", "neg_item"])