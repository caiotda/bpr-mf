import numpy as np


def average_precision_at_k(ranked_items, relevant_items, k):
    score = 0.0
    hits = 0

    for i, item in enumerate(ranked_items[:k], start=1):
        if item in relevant_items:
            hits += 1
            score += hits / i

    if len(relevant_items) == 0:
        return 0.0

    return score / min(len(relevant_items), k)


def compute_map_at_k(top_k, eval_users, test_pos, k):
    map_scores = []
    for user_idx, user_id in enumerate(eval_users):
        relevant_items = test_pos[user_id]
        if len(relevant_items) == 0:
            continue
        ap = average_precision_at_k(
            ranked_items=top_k[user_idx],
            relevant_items=relevant_items,
            k=k,
        )
        map_scores.append(ap)
    return float(np.mean(map_scores)) if map_scores else 0.0

def calculate_mrr(df):
    # We get the higher ranked relevant item for each user in each round.
    mrr_df = df.groupby("user").agg({"clicked_at": min}).reset_index()
    mrr_df["reciprocal_rank"] = 1 / (mrr_df["clicked_at"] + 1)
    mrr = mrr_df["reciprocal_rank"].mean()
    return mrr
