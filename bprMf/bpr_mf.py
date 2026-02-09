import abc
import torch
from torch import nn


import numpy as np
import pandas as pd

from bprMf.utils.learner import bpr_loss_with_reg, bpr_loss_with_reg_with_debiased_click
from bprMf.utils.data import create_bpr_dataloader
from bprMf.model import baseModel

from bprMf.evaluation import average_precision_at_k
from tqdm import trange
import gc


class bprMFBase(baseModel):
    def __init__(self, num_users, num_items, factors, reg_lambda, n_epochs, dev, lr):
        super().__init__()
        self.device = dev
        self.lr = lr
        self.user_emb = nn.Embedding(
            num_embeddings=num_users, embedding_dim=factors, device=dev
        )
        self.item_emb = nn.Embedding(
            num_embeddings=num_items, embedding_dim=factors, device=dev
        )
        self.n_users = num_users
        self.n_items = num_items
        nn.init.normal_(self.user_emb.weight, mean=0, std=0.01)
        nn.init.normal_(self.item_emb.weight, mean=0, std=0.01)
        self.reg_lambda = reg_lambda
        self.n_epochs = n_epochs

    @abc.abstractmethod
    def fit(self, train_df, debug=False):
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

    def evaluate(self, test_df, k=20):
        self.eval()
        map_scores = []
        candidates = torch.tensor(test_df["item"].unique(), device=self.device)
        with torch.no_grad():
            for user_id, user_df in test_df.groupby("user"):

                user_id = torch.tensor([user_id], device=self.device)
                hist = set(user_df.item.values)
                scores = self.forward(user_id, candidates)

                top_k_indices = torch.topk(scores, k=k).indices
                top_k_items = candidates[top_k_indices].cpu().numpy()
                ap = average_precision_at_k(
                    ranked_items=top_k_items, relevant_items=list(hist), k=k
                )
                map_scores.append(ap)
        self.train()

        return np.mean(map_scores).item()

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
    def __init__(self, num_users, num_items, factors, reg_lambda, n_epochs, dev, lr):
        super().__init__(num_users, num_items, factors, reg_lambda, n_epochs, dev, lr)

    def fit(self, train_df, debug=False):
        train_data_loader = create_bpr_dataloader(train_df, should_debias=False)
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
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


class bprMFWithClickDebiasing(bprMFBase):
    def __init__(self, num_users, num_items, factors, reg_lambda, n_epochs, dev, lr):
        super().__init__(num_users, num_items, factors, reg_lambda, n_epochs, dev, lr)

    def fit(self, train_df, debug=False):
        train_data_loader = create_bpr_dataloader(train_df, should_debias=True)
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
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
        # Handles possible OOM after training
        del optimizer
        del users_factors
        del positive_items_factors
        del negative_items_factors
        gc.collect()
        torch.cuda.empty_cache()
        return train_epoch_losses
