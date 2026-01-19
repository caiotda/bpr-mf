import pandas as pd
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
