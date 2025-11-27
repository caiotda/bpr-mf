import torch

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
    # Applies l2 regularization as per BPR-OPT.
    reg = reg_lambda * (
        user_factor.pow(2).sum(dim=1) +
        positive_item_factor.pow(2).sum(dim=1) +
        negative_item_factor.pow(2).sum(dim=1)
    ).mean()

    return loss + reg