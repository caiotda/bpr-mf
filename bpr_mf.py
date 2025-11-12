
import torch
from torch import nn
from torch.utils.data import Dataset

import numpy as np

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


class bprMFBase(nn.Module):
    def __init__(self):
        super().__init__()

    def fit(self, train_data_loader, debug=False):
        pass

    def forward(self, user, item):
        assert torch.all(user >= 0) and torch.all(user < self.user_emb.num_embeddings), "User index out of range"
        assert torch.all(item >= 0) and torch.all(item < self.item_emb.num_embeddings), "Item index out of range"

        user_emb = self.user_emb(user)
        item_emb = self.item_emb(item)
        dot = (user_emb * item_emb).sum(dim=1)
        return dot

    def predict(self, user, candidates, k=100):
        assert torch.all(user >= 0) and torch.all(user < self.user_emb.num_embeddings), "User index out of range"
        assert torch.all(candidates >= 0) and torch.all(candidates < self.item_emb.num_embeddings), "Candidate item indices out of range"
        items_list = candidates
        output = self.forward(user, items_list)
        scored_items = list(zip(candidates.tolist(), output.tolist()))
        results_ranked_by_model = sorted(scored_items, key=lambda l: l[1], reverse=True)[:k]
        items, scores = zip(*results_ranked_by_model) if results_ranked_by_model else ([], [])
        return list(items), list(scores)

    def predict_flat(self, user, candidates, k=100):
        prediction = self.predict(user, candidates, k)
        return prediction[0]

 
class bprMf(bprMFBase):
    def __init__(self, num_users, num_items, factors, reg_lambda, n_epochs):
        super().__init__()
        self.user_emb = nn.Embedding(num_embeddings=num_users, embedding_dim=factors)
        self.item_emb = nn.Embedding(num_embeddings=num_items, embedding_dim=factors)
        nn.init.normal_(self.user_emb.weight, mean=0, std=0.01)
        nn.init.normal_(self.item_emb.weight, mean=0, std=0.01)
        self.reg_lambda = reg_lambda
        self.n_epochs = n_epochs

    def fit(self, train_data_loader, optimizer, debug=False):
        train_epoch_losses = []
        self.train()
        for epoch in range(self.n_epochs):
            batch_losses = []
            for _, ((user_ids, positive_items_ids, negative_items_ids)) in enumerate(train_data_loader):
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
        self.train()
        return float(np.mean(test_losses))




class bprMFWithClickDebiasing(bprMFBase):
    def __init__(self, num_users, num_items, factors, reg_lambda, n_epochs):
        super().__init__()
        self.user_emb = nn.Embedding(num_embeddings=num_users, embedding_dim=factors)
        self.item_emb = nn.Embedding(num_embeddings=num_items, embedding_dim=factors)
        nn.init.normal_(self.user_emb.weight, mean=0, std=0.01)
        nn.init.normal_(self.item_emb.weight, mean=0, std=0.01)
        self.reg_lambda = reg_lambda
        self.n_epochs = n_epochs

    def fit(self, train_data_loader, optimizer, debug=False):
        train_epoch_losses = []
        self.train()
        for epoch in range(self.n_epochs):
            batch_losses = []
            for _, ((user_ids, positive_items_ids, negative_items_ids, clicked_positions)) in enumerate(train_data_loader):
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
            epoch_loss = np.mean(batch_losses) if batch_losses else 0.0
            train_epoch_losses.append(epoch_loss)
            if debug:
                print(f"Train epoch mean loss: {epoch_loss:>7f}; Epoch: {epoch+1}/{self.n_epochs}")
        return train_epoch_losses
    
    def evaluate(self, test_data_loader):
        self.eval()
        test_losses = []
        with torch.no_grad():
            for batch in test_data_loader:
                user_ids, pos_item_ids, neg_item_ids, clicked_positions = batch
                user_ids = user_ids.to(device)
                users_factors =self.user_emb(user_ids)
                pos_item_ids = pos_item_ids.to(device)
                neg_item_ids = neg_item_ids.to(device)
                clicked_positions = clicked_positions.to(device)

                pred_positive =self(user_ids, pos_item_ids)
                pred_negative =self(user_ids, neg_item_ids)

                positive_items_factors =self.item_emb(pos_item_ids)
                negative_items_factors =self.item_emb(neg_item_ids)
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
        avg_test_loss = sum(test_losses) / len(test_losses)
        self.train()
        return avg_test_loss

        


