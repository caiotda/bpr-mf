import numpy as np


def _compute_map_at_k(top_k, eval_users, test_pos, k):
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


def compute_map_at_k(train_df, test_df, rec, users, top_k_for_evaluation):
    def _prepare_eval_data(train_df, test_df, users, rec, top_k_for_evaluation):
        train_pos = train_df.groupby("user")["item"].apply(set)
        test_pos = test_df.groupby("user")["item"].apply(set)
        # Make sure to use users present in both, specially important in CVTT
        # scenario
        eval_users = sorted(set(test_pos.index) & set(train_pos.index))
        all_users_in_rec = users.cpu().numpy()

        user_to_idx = {user_id: idx for idx, user_id in enumerate(all_users_in_rec)}
        eval_indices = [user_to_idx[u] for u in eval_users]
        top_k = rec[eval_indices, :top_k_for_evaluation].cpu().numpy()

        return top_k, eval_users, test_pos

    top_k_eval, eval_users, test_pos = _prepare_eval_data(
        train_df, test_df, users, rec, top_k_for_evaluation
    )
    map_k = _compute_map_at_k(top_k_eval, eval_users, test_pos, top_k_for_evaluation)

    return map_k


def calculate_mmr(df):
    # We get the higher ranked relevant item for each user in each round.
    mrr_df = df.groupby("user").agg({"clicked_at": "min"}).reset_index()
    mrr_df["reciprocal_rank"] = 1 / (mrr_df["clicked_at"] + 1)
    mrr = mrr_df["reciprocal_rank"].mean()
    return mrr
