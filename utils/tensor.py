import torch


def create_id_to_idx_lookup_tensor(tensor):
    lookup = -1 + torch.zeros(tensor.max().item() + 1, dtype=torch.long)
    lookup[tensor] = torch.arange(len(tensor))
    return lookup