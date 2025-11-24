from collections import defaultdict
import numpy as np
import pandas as pd

from torch.utils.data import DataLoader, random_split
import torch

from bprMf.bpr_mf import bprMFDataloader
import time

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def build_positive_and_negatives_bpr_tensor(user_positives_tensor, n_users, n_items):

    user_ids, item_ids = torch.meshgrid(
        torch.arange(n_users, device=device),
        torch.arange(n_items, device=device),
        indexing='ij'
    )
    all_user_item_pairs = torch.stack([user_ids.reshape(-1), item_ids.reshape(-1)], dim=1)

    max_item = n_items + 1 
    all_pairs_1d = all_user_item_pairs[:, 0] * max_item + all_user_item_pairs[:, 1]
    indices_1d = user_positives_tensor[:, 0] * max_item + user_positives_tensor[:, 1]

    mask = ~torch.isin(all_pairs_1d, indices_1d)
    neg_examples = all_user_item_pairs[mask]
    neg_examples = torch.cat([
        neg_examples,
        torch.zeros((neg_examples.shape[0], 1), device=device,dtype=torch.int)
    ], dim=1)

    return torch.cat([neg_examples, user_positives_tensor], dim=0)


def generate_bpr_dataset(interactions_dataset, num_negatives=3):
    positive_interactions = interactions_dataset.loc[interactions_dataset["relevant"] == 1, ["user", "item", "relevant"]]
    n_users = interactions_dataset.user.max() + 1
    n_items = interactions_dataset.item.max() + 1
    positive_interactions_tensor = torch.tensor(positive_interactions.values, device=device)
    indices = positive_interactions_tensor[:, 0:2]
    train_tensor = build_positive_and_negatives_bpr_tensor(positive_interactions_tensor, n_users, n_items)

    user_ids = train_tensor[:, 0].long()
    item_ids = train_tensor[:, 1].long()
    labels   = train_tensor[:, 2].long()

    pos_mask = torch.zeros((n_users, n_items), dtype=torch.bool, device=train_tensor.device)
    pos_mask[user_ids[labels==1], item_ids[labels==1]] = True

    pos_users = user_ids[labels==1]
    pos_items = item_ids[labels==1]

    neg_samples = torch.randint(
        low=0,
        high=n_items,
        size=(len(pos_items), num_negatives),
        device=train_tensor.device
    )

    bad = pos_mask[pos_users.unsqueeze(1), neg_samples]

    while True:
        if not bad.any():
            break

        # re-draw only the bad negatives
        num_bad = bad.sum()
        neg_samples[bad] = torch.randint(
            0, n_items, size=(num_bad,), device=train_tensor.device
        )
        
        # recompute mask
        bad = pos_mask[pos_users.unsqueeze(1), neg_samples]
    
    neg_items = neg_samples.reshape(len(pos_items) * num_negatives, 1)
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
    start_time = time.time()
    print("Starting training process...")

    print("Creating training dataset...")
    dataset_start = time.time()
    train_data_loader = create_train_dataset(data, train_ratio)
    print(f"Training dataset created in {time.time() - dataset_start:.2f} seconds.")

    print("Initializing optimizer...")
    optimizer_start = time.time()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    print(f"Optimizer initialized in {time.time() - optimizer_start:.2f} seconds.")

    print("Fitting model...")
    fit_start = time.time()
    losses = model.fit(train_data_loader, optimizer)
    print(f"Model fit completed in {time.time() - fit_start:.2f} seconds.")

    print(f"Total training process completed in {time.time() - start_time:.2f} seconds.")
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