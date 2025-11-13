
import abc
import torch
from torch import nn
from torch.utils.data import Dataset

import numpy as np
import pandas as pd

from bpr_utils import bpr_loss_with_reg, bpr_loss_with_reg_with_debiased_click

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class bprMFDataloader(Dataset):
    def __init__(self, bpr_df):
        self.data = bpr_df
        self.users = torch.tensor(bpr_df["user"].values, dtype=torch.long)
        self.pos_items = torch.tensor(bpr_df["pos_item"].values, dtype=torch.long)
        self.neg_items = torch.tensor(bpr_df["neg_item"].values, dtype=torch.long)

    def __len__(self):
        return len(self.users)

    def __getitem__(self, idx):
        return self.users[idx], self.pos_items[idx], self.neg_items[idx]

class bprMFLClickDebiasingDataloader(Dataset):
    def __init__(self, bpr_df):
        self.data = bpr_df
        self.users = torch.tensor(bpr_df["user"].values, dtype=torch.long)
        self.pos_items = torch.tensor(bpr_df["pos_item"].values, dtype=torch.long)
        self.neg_items = torch.tensor(bpr_df["neg_item"].values, dtype=torch.long)
        self.click_position = torch.tensor(bpr_df["click_position"].values, dtype=torch.long)

    def __len__(self):
        return len(self.users)

    def __getitem__(self, idx):
        return (
            self.users[idx],
            self.pos_items[idx],
            self.neg_items[idx],
            self.click_position[idx]
        )

class bprMFBase(nn.Module, abc.ABC):
    def __init__(self, num_users, num_items, factors, reg_lambda, n_epochs):
        super().__init__()
        self.user_emb = nn.Embedding(num_embeddings=num_users, embedding_dim=factors)
        self.item_emb = nn.Embedding(num_embeddings=num_items, embedding_dim=factors)
        nn.init.normal_(self.user_emb.weight, mean=0, std=0.01)
        nn.init.normal_(self.item_emb.weight, mean=0, std=0.01)
        self.reg_lambda = reg_lambda
        self.n_epochs = n_epochs

    @abc.abstractmethod
    def fit(self, train_data_loader, optimizer, debug=False):
        pass

    def forward(self, user, item):
        assert torch.all(user >= 0) and torch.all(user < self.user_emb.num_embeddings), "User index out of range"
        assert torch.all(item >= 0) and torch.all(item < self.item_emb.num_embeddings), "Item index out of range"

        user_emb = self.user_emb(user)
        item_emb = self.item_emb(item)
        mult = user_emb @ item_emb.T
        return mult

    def predict(self, user, candidates, k=100):
        assert torch.all(user >= 0) and torch.all(user < self.user_emb.num_embeddings), "User index out of range"
        assert torch.all(candidates >= 0) and torch.all(candidates < self.item_emb.num_embeddings), "Candidate item indices out of range"
        items_list = candidates
        output = self.forward(user, items_list)
        # Sorts column-wise: each row contains the ranked recommendation
        scored_matrix, indices = output.sort(dim=1, descending=True)
        return indices[:, :k], scored_matrix[:, :k]
    
    def score(self, test_df, k=100, candidates=None):
        if candidates is None:
            items = test_df[["item"]].drop_duplicates()
        else:
            items = candidates
        users = test_df[["user"]].drop_duplicates()

        users_tensor = torch.tensor(users, device=device)
        items_tensor = torch.tensor(items, device=device)
        item_recs = self.predict(users_tensor, items_tensor, k)[0]

        scored_df = test_df.copy()
        predictions_series = pd.Series(item_recs.cpu().tolist(), index=test_df.index)
        scored_df[f"top_{k}_rec"] = predictions_series
        
        return scored_df

    def predict_flat(self, user, candidates, k=100):
        prediction = self.predict(user, candidates, k)
        return prediction[0]

class bprMf(bprMFBase):
    def __init__(self, num_users, num_items, factors, reg_lambda, n_epochs):
        super().__init__(num_users, num_items, factors, reg_lambda, n_epochs)

    def fit(self, train_data_loader, optimizer, debug=False):
        train_epoch_losses = []
        self.train()
        for epoch in range(self.n_epochs):
            batch_losses = []
            for _, (user_ids, positive_items_ids, negative_items_ids) in enumerate(train_data_loader):
                user_ids = user_ids.to(device)
                positive_items_ids = positive_items_ids.to(device)
                negative_items_ids = negative_items_ids.to(device)

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
                    self.reg_lambda
                )

                batch_losses.append(loss.detach().item())

                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
            epoch_loss = float(np.mean(batch_losses)) if batch_losses else 0.0
            train_epoch_losses.append(epoch_loss)
            if debug:
                print(f"Train epoch mean loss: {epoch_loss:>7f}; Epoch: {epoch+1}/{self.n_epochs}")
        return train_epoch_losses

    def evaluate(self, test_data_loader):
        self.eval()
        test_losses = []
        try:
            with torch.no_grad():
                for batch in test_data_loader:
                    user_ids, pos_item_ids, neg_item_ids = batch
                    user_ids = user_ids.to(device)
                    users_factors = self.user_emb(user_ids)
                    pos_item_ids = pos_item_ids.to(device)
                    neg_item_ids = neg_item_ids.to(device)

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
                        reg_lambda=self.reg_lambda
                    )
                    test_losses.append(loss.item())
        finally:
            self.train()
        return float(np.mean(test_losses)) if test_losses else 0.0




class bprMFWithClickDebiasing(bprMFBase):
    def __init__(self, num_users, num_items, factors, reg_lambda, n_epochs):
        super().__init__(num_users, num_items, factors, reg_lambda, n_epochs)

    def fit(self, train_data_loader, optimizer, debug=False):
        train_epoch_losses = []
        self.train()
        for epoch in range(self.n_epochs):
            batch_losses = []
            for _, (user_ids, positive_items_ids, negative_items_ids, clicked_positions) in enumerate(train_data_loader):
                user_ids = user_ids.to(device)

                positive_items_ids = positive_items_ids.to(device)
                negative_items_ids = negative_items_ids.to(device)
                clicked_positions = clicked_positions.to(device)

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
                    self.reg_lambda
                )

                batch_losses.append(loss.detach().item())

                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
            epoch_loss = float(np.mean(batch_losses)) if batch_losses else 0.0
            train_epoch_losses.append(epoch_loss)
            if debug:
                print(f"Train epoch mean loss: {epoch_loss:>7f}; Epoch: {epoch+1}/{self.n_epochs}")
        return train_epoch_losses
    
    def evaluate(self, test_data_loader):
        self.eval()
        test_losses = []
        try:
            with torch.no_grad():
                for batch in test_data_loader:
                    user_ids, pos_item_ids, neg_item_ids, clicked_positions = batch
                    user_ids = user_ids.to(device)
                    users_factors = self.user_emb(user_ids)
                    pos_item_ids = pos_item_ids.to(device)
                    neg_item_ids = neg_item_ids.to(device)
                    clicked_positions = clicked_positions.to(device)

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
                        self.reg_lambda
                    )
                    test_losses.append(loss.item())
            avg_test_loss = float(np.mean(test_losses))
        finally:
            self.train()
        return avg_test_loss

        


