import abc
import torch
from torch import nn


import numpy as np
import pandas as pd

from bprMf.utils.learner import bpr_loss_with_reg, bpr_loss_with_reg_with_debiased_click
from bprMf.utils.data import create_bpr_dataloader
from bprMf.utils.tensor import create_id_to_idx_lookup_tensor
from tqdm import trange


class bprMFBase(nn.Module, abc.ABC):
    def __init__(self, num_users, num_items, factors, reg_lambda, n_epochs, dev):
        super().__init__()
        self.device = dev
        self.user_emb = nn.Embedding(num_embeddings=num_users, embedding_dim=factors)
        self.item_emb = nn.Embedding(num_embeddings=num_items, embedding_dim=factors)
        self.n_users = num_users
        self.n_items = num_items
        nn.init.normal_(self.user_emb.weight, mean=0, std=0.01)
        nn.init.normal_(self.item_emb.weight, mean=0, std=0.01)
        self.reg_lambda = reg_lambda
        self.n_epochs = n_epochs

    @abc.abstractmethod
    def fit(self, train_df, debug=False, lr=1e-3):
        pass

    def forward(self, users, item):
        assert torch.all(users >= 0) and torch.all(
            users < self.user_emb.num_embeddings
        ), "Users index out of range"
        assert torch.all(item >= 0) and torch.all(
            item < self.item_emb.num_embeddings
        ), "Item index out of range"

        user_emb = self.user_emb(users)
        item_emb = self.item_emb(item)
        mult = (user_emb * item_emb).sum(dim=1)
        return mult

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

            # Por que ele faz .cpu?
            users_all = torch.cat(all_users, dim=0).cpu().unsqueeze(1).long()
            items_all = torch.cat(all_items, dim=0).cpu().unsqueeze(1).long()
            scores_all = torch.cat(all_scores, dim=0).cpu().unsqueeze(1)

            del all_users
            del all_items
            del all_scores

        return users_all, items_all, scores_all
    


    def recommend(self, users, candidates, k=100, mask=None):
        """
        Generate top-k recommendations for a batch of users from a set of candidate items.

        Args:
            users (torch.Tensor): A 1D tensor containing user IDs or indices for which recommendations are to be generated.
            candidates (torch.Tensor): A 1D tensor containing candidate item IDs or indices to consider for recommendation.
            k (int, optional): The number of top recommendations to return for each user. Defaults to 100.
            mask (torch.Tensor, optional): An optional mask tensor of the same shape as the prediction output, where positions with value 1 are considered valid for recommendation, and positions with value 0 are ignored (set to -inf in scores).

        Returns:
            tuple:
                candidate_ids (torch.Tensor): A 2D tensor of shape (num_users, k) containing the IDs or indices of the top-k recommended items for each user.
                scored_matrix (torch.Tensor): A 2D tensor of shape (num_users, k) containing the corresponding scores for the recommended items.

        """
        self.check_input_tensor_dimensions_for_prediction(users, candidates, mask)
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

    def check_input_tensor_dimensions_for_prediction(self, user, candidates, mask):
        assert user.dim() == 1, "User tensor must be 1-dimensional"
        if mask is not None:
            assert mask.dim() == 2, "Mask tensor must be 2-dimensional"
            assert mask.size(0) == user.size(
                0
            ), "Mask's first dimension must match user tensor size"
            assert mask.size(1) == candidates.size(
                0
            ), "Mask's second dimension must match candidates tensor size"
            assert torch.all(
                (mask == -1) | (mask == 1)
            ), "Mask values must be either -1 or 1"

        assert candidates.dim() == 1, "Candidates tensor must be 1-dimensional"
        assert torch.all(user >= 0) and torch.all(
            user < self.user_emb.num_embeddings
        ), "User index out of range"
        assert torch.all(candidates >= 0) and torch.all(
            candidates < self.item_emb.num_embeddings
        ), "Candidate item indices out of range"

    def score(self, test_df, k=100, candidates=None):
        if candidates is None:
            items = test_df["item"].drop_duplicates()
        else:
            items = candidates
        users = test_df["user"].drop_duplicates()

        users_tensor = torch.tensor(users, dtype=torch.long, device=self.device)
        items_tensor = torch.tensor(items, dtype=torch.long, device=self.device)
        item_recs, item_scores = self.recommend(
            users=users_tensor, candidates=items_tensor, k=k
        )

        scored_df = pd.DataFrame(
            zip(users_tensor.tolist(), item_recs.tolist(), item_scores.tolist()),
            columns=["user", "top_k_rec_id", "top_k_rec_score"],
        )

        return scored_df

    def predict_flat(self, user, candidates, k=100):
        prediction = self.predict(user, candidates, k)
        return prediction[0]


class bprMf(bprMFBase):
    def __init__(self, num_users, num_items, factors, reg_lambda, n_epochs, dev):
        super().__init__(num_users, num_items, factors, reg_lambda, n_epochs, dev)

    def fit(self, train_df, debug=False, lr=1e-3):
        train_data_loader = create_bpr_dataloader(train_df, should_debias=False)
        optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        train_epoch_losses = []
        self.train()
        for epoch in trange(self.n_epochs, desc="Epochs"):
            batch_losses = []
            for _, (user_ids, positive_items_ids, negative_items_ids) in enumerate(
                train_data_loader
            ):
                user_ids = user_ids.to(self.device)
                positive_items_ids = positive_items_ids.to(self.device)
                negative_items_ids = negative_items_ids.to(self.device)

                pred_positive = self.forward(user_ids, positive_items_ids)
                pred_negative = self.forward(user_ids, negative_items_ids)

                users_factors = self.user_emb(user_ids)
                positive_items_factors = self.item_emb(positive_items_ids)
                negative_items_factors = self.item_emb(negative_items_ids)

                loss = bpr_loss_with_reg(
                    pred_positive,
                    pred_negative,
                    users_factors,
                    positive_items_factors,
                    negative_items_factors,
                    self.reg_lambda,
                )

                batch_losses.append(loss.detach().item())

                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
            epoch_loss = float(np.mean(batch_losses)) if batch_losses else 0.0
            train_epoch_losses.append(epoch_loss)
            if debug:
                user_norm = users_factors.norm(dim=1)
                pos_item_norm = positive_items_factors.norm(dim=1)
                neg_item_norm = negative_items_factors.norm(dim=1)
                print("User norm", user_norm.mean().item(), user_norm.std().item())
                print(
                    "pos_item norm",
                    pos_item_norm.mean().item(),
                    pos_item_norm.std().item(),
                )
                print(
                    "neg_item norm",
                    neg_item_norm.mean().item(),
                    neg_item_norm.std().item(),
                )
                print(
                    f"Train epoch mean loss: {epoch_loss:>7f}; Epoch: {epoch+1}/{self.n_epochs}"
                )
        return train_epoch_losses

    def evaluate(self, test_df, k=20):
        self.eval()
        test_losses = []
        test_data_loader = create_bpr_dataloader(test_df, should_debias=False)
        try:
            with torch.no_grad():
                for batch in test_data_loader:
                    user_ids, pos_item_ids, neg_item_ids = batch
                    user_ids = user_ids.to(self.device)
                    users_factors = self.user_emb(user_ids)
                    pos_item_ids = pos_item_ids.to(self.device)
                    neg_item_ids = neg_item_ids.to(self.device)

                    pred_positive = self(user_ids, pos_item_ids)
                    pred_negative = self(user_ids, neg_item_ids)

                    positive_items_factors = self.item_emb(pos_item_ids)
                    negative_items_factors = self.item_emb(neg_item_ids)
                    loss = bpr_loss_with_reg(
                        pred_positive,
                        pred_negative,
                        users_factors,
                        positive_items_factors,
                        negative_items_factors,
                        reg_lambda=self.reg_lambda,
                    )
                    test_losses.append(loss.item())
                avg_test_loss = float(np.mean(test_losses)) if test_losses else 0.0
        finally:
            self.train()
        return avg_test_loss


class bprMFWithClickDebiasing(bprMFBase):
    def __init__(self, num_users, num_items, factors, reg_lambda, n_epochs, dev):
        super().__init__(num_users, num_items, factors, reg_lambda, n_epochs, dev)

    def fit(self, train_df, debug=False, lr=1e-3):
        train_data_loader = create_bpr_dataloader(train_df, should_debias=True)
        optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        train_epoch_losses = []
        self.train()
        for epoch in range(self.n_epochs):
            batch_losses = []
            for _, (
                user_ids,
                positive_items_ids,
                negative_items_ids,
                clicked_positions,
            ) in enumerate(train_data_loader):
                user_ids = user_ids.to(self.device)

                positive_items_ids = positive_items_ids.to(self.device)
                negative_items_ids = negative_items_ids.to(self.device)
                clicked_positions = clicked_positions.to(self.device)

                pred_positive = self.forward(user_ids, positive_items_ids)
                pred_negative = self.forward(user_ids, negative_items_ids)

                users_factors = self.user_emb(user_ids)
                positive_items_factors = self.item_emb(positive_items_ids)
                negative_items_factors = self.item_emb(negative_items_ids)

                loss = bpr_loss_with_reg_with_debiased_click(
                    pred_positive,
                    pred_negative,
                    clicked_positions,
                    users_factors,
                    positive_items_factors,
                    negative_items_factors,
                    self.reg_lambda,
                )

                batch_losses.append(loss.detach().item())

                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
            epoch_loss = float(np.mean(batch_losses)) if batch_losses else 0.0
            train_epoch_losses.append(epoch_loss)
            if debug:
                print(
                    f"Train epoch mean loss: {epoch_loss:>7f}; Epoch: {epoch+1}/{self.n_epochs}"
                )
        return train_epoch_losses

    def evaluate(self, test_df, k=20):
        self.eval()
        test_losses = []
        test_data_loader = create_bpr_dataloader(test_df, should_debias=True)
        try:
            with torch.no_grad():
                for batch in test_data_loader:
                    user_ids, pos_item_ids, neg_item_ids, clicked_positions = batch
                    user_ids = user_ids.to(self.device)
                    users_factors = self.user_emb(user_ids)
                    pos_item_ids = pos_item_ids.to(self.device)
                    neg_item_ids = neg_item_ids.to(self.device)
                    clicked_positions = clicked_positions.to(self.device)

                    pred_positive = self(user_ids, pos_item_ids)
                    pred_negative = self(user_ids, neg_item_ids)

                    positive_items_factors = self.item_emb(pos_item_ids)
                    negative_items_factors = self.item_emb(neg_item_ids)
                    loss = bpr_loss_with_reg_with_debiased_click(
                        pred_positive,
                        pred_negative,
                        clicked_positions,
                        users_factors,
                        positive_items_factors,
                        negative_items_factors,
                        self.reg_lambda,
                    )
                    test_losses.append(loss.item())
            avg_test_loss = float(np.mean(test_losses))
        finally:
            self.train()
        return avg_test_loss
