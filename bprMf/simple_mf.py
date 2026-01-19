from torch import nn
from torch.utils.data import Dataset, DataLoader
import torch

import pandas as pd


class MF(nn.Module):
    def __init__(self, num_users, num_items, factors):
        super().__init__()
        self.user_emb = nn.Embedding(num_embeddings=num_users, embedding_dim=factors)
        self.item_emb = nn.Embedding(num_embeddings=num_items, embedding_dim=factors)
    def forward(self, user, item):
        user_emb = self.user_emb(user)
        item_emb = self.item_emb(item)
        dot =  (user_emb * item_emb).sum(dim=1)
        return torch.sigmoid(dot) * 4 + 1

class MFDataLoader(Dataset):
    def __init__(self, file_path):
        self.target_column = 'rating'
        self._columns = ['user', 'item', self.target_column]
        self.data = pd.read_csv(file_path)[self._columns]
        self.n_users = self.data['user'].nunique()
        self.n_items = self.data['item'].nunique()

        unique_user_ids = self.data['user'].unique()
        unique_item_ids = self.data['item'].unique()
        self.user_id_map = {old_id: new_id for new_id, old_id in enumerate(sorted(unique_user_ids))}
        self.item_id_map = {old_id: new_id for new_id, old_id in enumerate(sorted(unique_item_ids))}
        self.data['user'] = self.data['user'].map(self.user_id_map)
        self.data['item'] = self.data['item'].map(self.item_id_map)



    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]

        user = int(row['user'])
        item = int(row['item'])
        label = float(row[self.target_column])
        return (user, item), label
