
import torch
from torch import nn
from torch.utils.data import Dataset


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

        assert torch.all(user >= 0) and torch.all(user < self.user_emb.num_embeddings), "User index out of range"
        assert torch.all(item >= 0) and torch.all(item < self.item_emb.num_embeddings), "Item index out of range"
        
        user_emb = self.user_emb(user)
        item_emb = self.item_emb(item)
        dot =  (user_emb * item_emb).sum(dim=1)
        # We avoid applying sigmoid here as the BPR paper assumes that 
        # the predicted score is a simple dot product
        return dot
    
def get_click_propensity(clicks_positions):
    # Function that penalizes clicks on top ranked positions. This
    # serves as a debiasing approach for our model.
    return torch.log2(1 + clicks_positions)

def bpr_loss_with_reg_with_debiased_click(
            positive_item_scores,
            negative_item_scores,
            positive_items_positions,
            user_factor,
            positive_item_factor,
            negative_item_factor,
            reg_lambda=1e-4):
    w_u_i = get_click_propensity(positive_items_positions)
    loss = - w_u_i * torch.log(torch.sigmoid(positive_item_scores - negative_item_scores) + 1e-8)
    loss = loss.mean()
    batch_size = positive_item_scores.size(0)
    # Applies l2 regularization as per BPR-OPT.
    reg = reg_lambda * (
        user_factor.norm(2).pow(2) +
        positive_item_factor.norm(2).pow(2) +
        negative_item_factor.norm(2).pow(2)
    ) / batch_size

    return loss + reg


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

    return loss + reg




def bpr_train_with_debiasing(train_data_loader, test_data_loader, model, bpr_loss, optimizer, reg_lambda = 1e-4, n_epochs=10, debug=False):
    batch_losses = [] 
    train_epoch_losses = []
    test_epoch_losses = []
    model.train()
    for epoch in range(n_epochs):

        epoch_loss = []
        for batch, ((user_ids, positive_items_ids, negative_items_ids, clicked_positions)) in enumerate(train_data_loader):
            user_ids = user_ids.to(device)

            positive_items_ids = positive_items_ids.to(device)
            negative_items_ids = negative_items_ids.to(device)
            clicked_positions = clicked_positions.to(device)

            pred_positive = model(user_ids, positive_items_ids)
            pred_negative = model(user_ids, negative_items_ids)


            users_factors = model.user_emb(user_ids)
            positive_items_factors = model.item_emb(positive_items_ids)
            negative_items_factors = model.item_emb(negative_items_ids)



            loss = bpr_loss(
                pred_positive,
                pred_negative,
                clicked_positions,
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
        train_epoch_loss = sum(epoch_loss) / len(epoch_loss)
        train_epoch_losses.append(train_epoch_loss)
        test_epoch_loss = bpr_eval_debiasing(test_data_loader, model, bpr_loss, reg_lambda)
        test_epoch_losses.append(test_epoch_loss)
        if debug:
            print(f"Train epoch mean loss: {train_epoch_loss:>7f};\n Test epoch mean loss: {test_epoch_loss:>7f}; Epoch: {epoch+1}/{n_epochs}")
    return train_epoch_losses, test_epoch_losses
        
def bpr_eval_debiasing(dataloader, model, bpr_loss, reg_lambda=1e-4):
    model.eval()
    test_losses = []
    with torch.no_grad():
        for batch in dataloader:
            user_ids, pos_item_ids, neg_item_ids, clicked_positions = batch
            user_ids = user_ids.to(device)
            users_factors = model.user_emb(user_ids)
            pos_item_ids = pos_item_ids.to(device)
            neg_item_ids = neg_item_ids.to(device)
            clicked_positions = clicked_positions.to(device)

            pred_positive = model(user_ids, pos_item_ids)
            pred_negative = model(user_ids, neg_item_ids)

            positive_items_factors = model.item_emb(pos_item_ids)
            negative_items_factors = model.item_emb(neg_item_ids)
            loss = bpr_loss(
                pred_positive,
                pred_negative,
                clicked_positions,
                users_factors,
                positive_items_factors,
                negative_items_factors,
                reg_lambda
            )
            test_losses.append(loss.item())
    avg_test_loss = sum(test_losses) / len(test_losses)
    model.train()
    return avg_test_loss


def bpr_eval(dataloader, model, bpr_loss, reg_lambda=1e-4):
    model.eval()
    test_losses = []
    with torch.no_grad():
        for batch in dataloader:
            user_ids, pos_item_ids, neg_item_ids = batch
            user_ids = user_ids.to(device)
            users_factors = model.user_emb(user_ids)
            pos_item_ids = pos_item_ids.to(device)
            neg_item_ids = neg_item_ids.to(device)

            pred_positive = model(user_ids, pos_item_ids)
            pred_negative = model(user_ids, neg_item_ids)

            positive_items_factors = model.item_emb(pos_item_ids)
            negative_items_factors = model.item_emb(neg_item_ids)
            loss = bpr_loss(pred_positive, pred_negative, users_factors, positive_items_factors, negative_items_factors, reg_lambda=reg_lambda)
            test_losses.append(loss.item())
    avg_test_loss = sum(test_losses) / len(test_losses)
    model.train()
    return avg_test_loss


def bpr_train(train_data_loader, test_data_loader, model, bpr_loss, optimizer, reg_lambda = 1e-4, n_epochs=10, debug=False):
    batch_losses = [] 
    train_epoch_losses = []
    test_epoch_losses = []
    model.train()
    for epoch in range(n_epochs):

        epoch_loss = []
        for batch, ((user_ids, positive_items_ids, negative_items_ids)) in enumerate(train_data_loader):
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
        train_epoch_losses.append(epoch_loss)
        test_epoch_loss = bpr_eval(test_data_loader, model, bpr_loss, reg_lambda)
        test_epoch_losses.append(test_epoch_loss)
        if debug:
            print(f"Train epoch mean loss: {epoch_loss:>7f};\n Test epoch mean loss: {test_epoch_loss:>7f}; Epoch: {epoch+1}/{n_epochs}")
    return train_epoch_losses, test_epoch_losses
        
