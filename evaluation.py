import torch
import pandas as pd
import numpy as np

class Evaluate():
    def __init__(self, model, test_set_loader, interaction_data, k=20):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self._df = test_set_loader.dataset.data
        self.model = model
        self.interaction = interaction_data
        self.k = k
        self.device = device
        self.test_set = self._prepare_test_set()

    def _prepare_test_set(self):
        evaluation_set = pd.merge(
            self._df[["user", "pos_item"]].drop_duplicates(),
            self.interaction[["user", "item", "relevant"]],
            left_on=["user", "pos_item"],
            right_on=["user", "item"]
        )[["user", "item", "relevant"]]


        relevant_set =  (evaluation_set[evaluation_set["relevant"] == 1] 
            .groupby("user").agg({"item": list}) 
            .rename(columns={"item": "relevant_items"}) 
        ).reset_index()
        candidate_items = evaluation_set.item.unique().astype(np.int64)
        candidates = torch.tensor(candidate_items, dtype=torch.long, device=self.device)
        self.candidates_for_evaluation = candidates
        for k in [1, 5, 10, 20, 25]:
            relevant_set[f"top_{k}_rec"] = relevant_set.apply(
            lambda row: self.model.predict(
                user=torch.tensor(data=row["user"], device=self.device),
                candidates=self.candidates_for_evaluation,
                k=k
            )[0],
            axis=1
            )
        return relevant_set.set_index("user")
    
    def precision(self, predicted, true):
        s_true = set(true)
        s_predicted = set(predicted)
        inter = s_predicted.intersection(s_true)
        return len(inter) / len(s_true)

    def precision_at_k(self,predicted, true, k):
        predicted_cut = predicted[:k]
        true_cut = true[:k]
        return self.precision(predicted_cut, true_cut)


    def average_precision_at_k(self,predicted, true):
        ap = []
        for k_i in range(1, self.k+1):
            ap.append(self.precision_at_k(predicted, true, k_i))
        return np.mean(ap)



    def MAP_at_k(self):
        map_k = self.test_set.apply(lambda r: self.average_precision_at_k(r[f"top_{self.k}_rec"], r["relevant_items"]), axis=1).mean().item()
        return map_k


