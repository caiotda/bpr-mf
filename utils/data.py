import torch
from torch.utils.data import Dataset, DataLoader


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class bprMFDataloader(Dataset):
    def __init__(self, bpr_tensor):
        self.users = bpr_tensor[:, 0]
        self.pos_items = bpr_tensor[:, 1]
        self.neg_items = bpr_tensor[:, 2]

    def __len__(self):
        return len(self.users)

    def __getitem__(self, idx):
        return self.users[idx], self.pos_items[idx], self.neg_items[idx]

class bprMFLClickDebiasingDataloader(Dataset):
    def __init__(self, bpr_tensor):
        self.users = bpr_tensor[:, 0]
        self.pos_items = bpr_tensor[:, 1]
        self.neg_items = bpr_tensor[:, 2]
        self.click_position = bpr_tensor[:, 3]

    def __len__(self):
        return len(self.users)

    def __getitem__(self, idx):
        return (
            self.users[idx],
            self.pos_items[idx],
            self.neg_items[idx],
            self.click_position[idx]
        )


def create_bpr_dataloader(df, should_debias=False):
    bpr_dataset = generate_bpr_triplets(df, num_negatives=5, use_click_debiasing=should_debias)
    
    if should_debias:
        data_bpr = bprMFLClickDebiasingDataloader(bpr_dataset)
    else:
        data_bpr = bprMFDataloader(bpr_dataset)

    return DataLoader(data_bpr, batch_size=256, shuffle=True)




def generate_bpr_triplets(interactions_dataset, num_negatives=3, use_click_debiasing=False):
    """
    Generates triplets of user, positive item, and negative item for Bayesian Personalized Ranking (BPR) training.
    Args:
        interactions_dataset (pd.DataFrame): A pandas DataFrame containing user-item interaction data. 
            It must include the columns "user", "item", and "relevant". If `use_click_debiasing` is True, 
            it should also include the "click" column.
        num_negatives (int, optional): The number of negative samples to generate for each positive interaction. 
            Defaults to 3.
        use_click_debiasing (bool, optional): Whether to include the "click" column in the interactions dataset 
            for debiasing purposes. Defaults to False.
    Returns:
        torch.Tensor: A tensor of shape `(num_positive_interactions * num_negatives, len(interactions_cols))` 
            containing the generated triplets. Each row represents a triplet in the format 
            `[user, positive_item, negative_item]`.
    Notes:
        - The function ensures that the negative items sampled for each user are not part of their positive 
          interactions.
        - The `interactions_dataset` should have integer-encoded "user" and "item" columns, where users and 
          items are indexed from 0 to `n_users - 1` and `n_items - 1`, respectively.
        - The function uses PyTorch tensors for efficient computation on GPU if available.
    """
    if use_click_debiasing:
        interactions_cols = ["user", "item", "clicked_at", "relevant"]
    else:
        interactions_cols = ["user", "item", "relevant"]

    n_users = interactions_dataset.user.max() + 1
    n_items = interactions_dataset.item.max() + 1

    positive_interactions = interactions_dataset.loc[interactions_dataset["relevant"] == 1, interactions_cols].astype(int)
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
    
    # Remove only the last tuple.
    cut_off = len(interactions_cols) - 1
    indices = positive_interactions_tensor[:, 0:cut_off]
    repeated_positives = indices.repeat_interleave(num_negatives, dim=0)
    triplets = torch.concat([repeated_positives, neg_items], dim=1)

    return triplets