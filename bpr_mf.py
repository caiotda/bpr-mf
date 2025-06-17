
import torch
from torch import nn
from torch.utils.data import Dataset


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class bprMFDataloader(Dataset):
    def __init__(self, bpr_df):
        self.users = torch.tensor(bpr_df["user"].values, dtype=torch.long)
        self.pos_items = torch.tensor(bpr_df["pos_item"].values, dtype=torch.long)
        self.neg_items = torch.tensor(bpr_df["neg_item"].values, dtype=torch.long)

    def __len__(self):
        return len(self.users)

    def __getitem__(self, idx):
        return self.users[idx], self.pos_items[idx], self.neg_items[idx]


class bprMF(nn.Module):
    def __init__(self, num_users, num_items, factors):
        super().__init__()
        self.user_emb = nn.Embedding(num_embeddings=num_users, embedding_dim=factors)
        self.item_emb = nn.Embedding(num_embeddings=num_items, embedding_dim=factors)
        # BPR paper assumes that parameters follow a normal distribution. They don´t mention
        # what the std used for the gaussian, just that its a small value. ARbitrarilly, we'll
        # use 1e-2.
        nn.init.normal_(self.user_emb.weight, mean=0, std=0.01)
        nn.init.normal_(self.item_emb.weight, mean=0, std=0.01)
    def forward(self, user, item):
        user_emb = self.user_emb(user)
        item_emb = self.item_emb(item)
        dot =  (user_emb * item_emb).sum(dim=1)
        # We avoid applying sigmoid here as the BPR paper assumes that 
        # the predicted score is a simple dot product
        return dot


def bpr_loss_with_reg(
            positive_item_scores,
            negative_item_scores,
            user_factor,
            positive_item_factor,
            negative_item_factor,
            reg_lambda=1e-4):
    loss = -torch.log(torch.sigmoid(positive_item_scores - negative_item_scores) + 1e-8).mean()
    batch_size = positive_item_scores.size(0)
    # Applies l2 regularization as per BPR-OPT.
    reg = reg_lambda * (
        user_factor.norm(2).pow(2) +
        positive_item_factor.norm(2).pow(2) +
        negative_item_factor.norm(2).pow(2)
    ) / batch_size


def bpr_train(dataloader, model, bpr_loss, optimizer, reg_lambda = 1e-4, n_epochs=10):
    batch_losses = [] 
    epoch_losses = []
    model.train()
    for epoch in range(n_epochs):

        epoch_loss = []
        for batch, ((user_ids, positive_items_ids, negative_items_ids)) in enumerate(dataloader):
            user_ids = user_ids.to(device)

            positive_items_ids = positive_items_ids.to(device)
            negative_items_ids = negative_items_ids.to(device)

            pred_positive = model(user_ids, positive_items_ids)
            pred_negative = model(user_ids, negative_items_ids)


            users_factors = model.user_emb(user_ids)
            positive_items_factors = model.item_emb(positive_items_ids)
            negative_items_factors = model.item_emb(negative_items_ids)


            loss = bpr_loss(
                pred_positive,
                pred_negative,
                users_factors,
                positive_items_factors,
                negative_items_factors,
                reg_lambda
            )

            epoch_loss.append(loss.detach().item())
            batch_losses.append(loss.detach().item())

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
        epoch_loss = sum(epoch_loss) / len(epoch_loss)
        epoch_losses.append(epoch_loss)
        print(f"epoch mean loss: {epoch_loss:>7f}; Epoch: {epoch+1}/{n_epochs}")
    return batch_losses, epoch_losses
        
