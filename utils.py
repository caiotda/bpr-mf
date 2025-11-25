from collections import defaultdict
import numpy as np
import pandas as pd

from torch.utils.data import DataLoader, random_split
import torch

from bprMf.bpr_mf import bprMFDataloader

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def generate_bpr_dataset(interactions_dataset, num_negatives=3):

    n_users = interactions_dataset.user.max() + 1
    n_items = interactions_dataset.item.max() + 1

    positive_interactions = interactions_dataset.loc[interactions_dataset["relevant"] == 1, ["user", "item", "relevant"]]
    positive_interactions_tensor = torch.tensor(positive_interactions.values, device=device)
    

    pos_users = positive_interactions_tensor[:, 0].long()
    pos_items = positive_interactions_tensor[:, 1].long()

    # We flag each positive user x item pair
    pos_mask = torch.zeros((n_users, n_items), dtype=torch.bool, device=positive_interactions_tensor.device)
    pos_mask[pos_users, pos_items] = True

    # Create a sample of size pos_items x num_negatives. With this,
    # We can sample exactly num_negatives for each pos_item. Using
    # pos_mask above, we assure that the sampled negatives are 
    # attributed exclusively to positive items.
    neg_samples = torch.randint(
        low=0,
        high=n_items,
        size=(len(pos_items), num_negatives),
        device=positive_interactions_tensor.device
    )

    # Ideally, this should be empty: all of the sampled negatives
    # are not present in pos_mask.
    bad = pos_mask[pos_users.unsqueeze(1), neg_samples]

    # Finally, we re-sample until the bad tensor is empty.
    while True:
        if not bad.any():
            break

        # re-draw only the bad negatives
        num_bad = bad.sum()
        neg_samples[bad] = torch.randint(
            low=0,
            high=n_items,
            size=(num_bad,),
            device=positive_interactions_tensor.device
        )
        
        # recompute mask
        bad = pos_mask[pos_users.unsqueeze(1), neg_samples]
    
    neg_items = neg_samples.reshape(len(pos_items) * num_negatives, 1)
    
    indices = positive_interactions_tensor[:, 0:2]
    repeated_positives = indices.repeat_interleave(num_negatives, dim=0)
    triplets = torch.concat([repeated_positives, neg_items], dim=1)

    return triplets

def create_train_dataset(data, train_ratio=1.0):
    bpr_dataset = generate_bpr_dataset(data, num_negatives=5)
    data_bpr = bprMFDataloader(bpr_dataset)


    train_len = int(train_ratio * len(data_bpr))
    test_len = len(data_bpr) - train_len


    train_data, _ = random_split(data_bpr, [train_len, test_len])



    return DataLoader(train_data, batch_size=1024, shuffle=True)


def train(model, data, train_ratio=1.0):
    train_data_loader = create_train_dataset(data, train_ratio)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    losses = model.fit(train_data_loader, optimizer)

    return model, losses
    

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
    # Assuming that the items are zero-indexed and continuous.
    n_items = int(interactions_dataset["item"].max()) + 1
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