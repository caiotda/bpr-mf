import torch
from torch.utils.data import Dataset, DataLoader


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


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
            self.click_position[idx],
        )


def create_bpr_dataloader(df, num_negatives, batch_size, should_debias=False):
    bpr_dataset = generate_bpr_triplets(
        df, num_negatives=num_negatives, use_click_debiasing=should_debias
    )

    if should_debias:
        data_bpr = bprMFLClickDebiasingDataloader(bpr_dataset)
    else:
        data_bpr = bprMFDataloader(bpr_dataset)

    return DataLoader(data_bpr, batch_size=batch_size, shuffle=True)


def generate_bpr_triplets(
    interactions_dataset, num_negatives=3, use_click_debiasing=False
):
    """
    Generates (user, positive_item, negative_item) triplets for BPR training.
    Negative items are sampled randomly from unobserved user-item pairs.

    Args:
        interactions_dataset: DataFrame of positive interactions with columns "user", "item",
            and optionally "clicked_at" if use_click_debiasing is True.
        num_negatives: Number of negative samples per positive interaction. Defaults to 3.
        use_click_debiasing: Whether to include "clicked_at" in the triplets. Defaults to False.

    Returns:
        Tensor of shape (num_interactions * num_negatives).
    """
    if use_click_debiasing:
        interactions_cols = ["user", "item", "clicked_at"]
    else:
        interactions_cols = ["user", "item"]

    n_users = interactions_dataset.user.max() + 1
    n_items = interactions_dataset.item.max() + 1

    positive_interactions = interactions_dataset[interactions_cols].astype(int)
    positive_interactions_tensor = torch.tensor(
        positive_interactions.values, device=device
    )

    users = positive_interactions_tensor[:, 0].long()
    pos_items = positive_interactions_tensor[:, 1].long()

    pos_mask = torch.zeros(
        (n_users, n_items), dtype=torch.bool, device=positive_interactions_tensor.device
    )
    pos_mask[users, pos_items] = True

    neg_samples = torch.randint(
        low=0,
        high=n_items,
        size=(len(pos_items), num_negatives),
        device=positive_interactions_tensor.device,
    )

    # Ideally, this should be empty: all of the sampled negatives
    # are not present in pos_mask.
    bad = pos_mask[users.unsqueeze(1), neg_samples]

    # Finally, we re-sample until the bad tensor is empty.
    while bad.any():
        # re-draw only the bad negatives
        num_bad = bad.sum()
        neg_samples[bad] = torch.randint(
            low=0,
            high=n_items,
            size=(num_bad,),
            device=positive_interactions_tensor.device,
        )

        # recompute mask
        bad = pos_mask[users.unsqueeze(1), neg_samples]

    neg_items = neg_samples.reshape(len(pos_items) * num_negatives, 1)

    cut_off = len(interactions_cols) - 1
    indices = positive_interactions_tensor[:, 0:cut_off]
    repeated_positives = indices.repeat_interleave(num_negatives, dim=0)
    triplets = torch.concat([repeated_positives, neg_items], dim=1)

    return triplets
