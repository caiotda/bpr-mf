import abc
import torch
from torch import nn


import numpy as np
import pandas as pd

from bprMf.utils.learner import bpr_loss_with_reg, bpr_loss_with_reg_with_debiased_click
from bprMf.utils.data import create_bpr_dataloader
from bprMf.utils.tensor import create_id_to_idx_lookup_tensor


from bprMf.evaluation import average_precision_at_k
from tqdm import trange
import gc


class baseModel(nn.Module, abc.ABC):
    def __init__(self):
        super().__init__()

    @abc.abstractmethod
    def fit(self, train_df, debug):
        pass

    @abc.abstractmethod
    def forward(self, users, items):
        pass

    def predict(self, users, candidates, batch_size=1_000_000):
        """
        Predict scores for all user-candidate item pairs.

        Args:
            users (Tensor): A 1D tensor of user IDs or indices.
            candidates (Tensor): A 1D tensor of candidate item IDs or indices.
            batch_size (int, optional): Number of user-item pairs to process per batch.

        Returns:
            users_all (Tensor): 2D tensor of user IDs for each scored pair, shape (num_pairs, 1).
            items_all (Tensor): 2D tensor of item IDs for each scored pair, shape (num_pairs, 1).
            scores_all (Tensor): 2D tensor of scores for each user-item pair, shape (num_pairs, 1).
        """

        user_grid, item_grid = torch.meshgrid(users, candidates, indexing="ij")
        combinations = torch.stack(
            [user_grid.reshape(-1), item_grid.reshape(-1)], dim=1
        )

        all_users = []
        all_items = []
        all_scores = []

        with torch.no_grad():
            for batch_start in range(0, len(combinations), batch_size):
                batch_end = min(batch_start + batch_size, len(combinations))
                users_batch = combinations[batch_start:batch_end, 0].to(self.device)
                items_batch = combinations[batch_start:batch_end, 1].to(self.device)
                scores_batch = self.forward(users_batch, items_batch)
                all_users.append(users_batch)
                all_items.append(items_batch)
                all_scores.append(scores_batch)

            users_all = torch.cat(all_users, dim=0).cpu().unsqueeze(1).long()
            items_all = torch.cat(all_items, dim=0).cpu().unsqueeze(1).long()
            scores_all = torch.cat(all_scores, dim=0).cpu().unsqueeze(1)

            del all_users
            del all_items
            del all_scores

        return users_all, items_all, scores_all

    def recommend(self, users, k, candidates, mask):

        n_users = len(users)
        n_candidates = len(candidates)

        users_stacked, items_stacked, scores = self.predict(users, candidates)

        user_id_to_idx_lookup = create_id_to_idx_lookup_tensor(users)
        item_id_to_idx_lookup = create_id_to_idx_lookup_tensor(candidates)

        user_idx = user_id_to_idx_lookup[users_stacked]
        item_idx = item_id_to_idx_lookup[items_stacked]

        prediction_matrix = -1 * torch.inf * torch.ones(size=(n_users, n_candidates))
        prediction_matrix[user_idx, item_idx] = scores

        if mask is not None:
            output = torch.where(mask == 1, prediction_matrix, float("-inf"))
        else:
            output = prediction_matrix
        # Sorts column-wise: each row contains the ranked recommendation
        scored_matrix, idx_matrix = output.sort(dim=1, descending=True)
        # Map indices back to actual candidate item IDs
        candidate_ids = candidates[idx_matrix[:, :k]]
        return candidate_ids, scored_matrix[:, :k]
