
import abc
import torch
from torch import nn
from torch.utils.data import Dataset

import numpy as np

from bprMf.bpr_utils import bpr_loss_with_reg, bpr_loss_with_reg_with_debiased_click

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
    def __init__(self, bpr_df):
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

    def predict(self, user, candidates, mask=None, k=100):
        """
        Predict top-k recommended items for a batch of users.

        Args:
            user (Tensor): A batch of users being scored.
            candidates (Tensor): A tensor of items to be scored.
            mask (Tensor, optional): A tensor indicating item-user pairs to ignore 
                (e.g., items already recommended to the users). Defaults to None.
            k (int, optional): The number of top items to return. Defaults to 100.

        Returns:
            Tuple[Tensor, Tensor]: A tuple containing:
                - candidate_ids (Tensor): The top-k recommended item IDs for each user.
                - scored_matrix (Tensor): The scores of the top-k items for each user.
        """
        self.check_input_tensor_dimensions_for_prediction(user, candidates, mask)
        items_list = candidates
        raw_prediction = self.forward(user, items_list)
        if mask is not None:
            output = torch.where(mask == 1, raw_prediction, float('-inf'))
        else:
            output = raw_prediction
        # Sorts column-wise: each row contains the ranked recommendation
        scored_matrix, indices = output.sort(dim=1, descending=True)
        # Map indices back to actual candidate item IDs
        candidate_ids = candidates[indices[:, :k]]
        return candidate_ids, scored_matrix[:, :k]

    def check_input_tensor_dimensions_for_prediction(self, user, candidates, mask):
        assert user.dim() == 1, "User tensor must be 1-dimensional"
        if mask is not None:
            assert mask.dim() == 2, "Mask tensor must be 2-dimensional"
            assert mask.size(0) == user.size(0), "Mask's first dimension must match user tensor size"
            assert mask.size(1) == candidates.size(0), "Mask's second dimension must match candidates tensor size"
            assert torch.all((mask == -1) | (mask == 1)), "Mask values must be either -1 or 1"

        assert candidates.dim() == 1, "Candidates tensor must be 1-dimensional"
        assert torch.all(user >= 0) and torch.all(user < self.user_emb.num_embeddings), "User index out of range"
        assert torch.all(candidates >= 0) and torch.all(candidates < self.item_emb.num_embeddings), "Candidate item indices out of range"
    
    def score(self, test_df, k=100, candidates=None):
        if candidates is None:
            items = test_df["item"].drop_duplicates()
        else:
            items = candidates
        users = test_df["user"].drop_duplicates()

        users_tensor = torch.tensor(users, dtype=torch.long, device=device)
        items_tensor = torch.tensor(items, dtype=torch.long, device=device)
        item_recs = self.predict(users_tensor, items_tensor, k)[0]

        scored_df = test_df.copy()

        user_to_pred = dict(zip(users.values, item_recs.cpu().tolist()))
        scored_df[f"top_{k}_rec"] = scored_df["user"].map(user_to_pred)
        
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

        


