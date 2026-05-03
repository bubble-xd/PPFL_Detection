from __future__ import annotations

from typing import Iterable

from sklearn.metrics import f1_score


def compute_detection_f1(
    predicted_ids: Iterable[int],
    malicious_ids: Iterable[int],
    num_clients: int,
) -> float:
    predicted_set = set(int(client_id) for client_id in predicted_ids)
    malicious_set = set(int(client_id) for client_id in malicious_ids)
    y_true = [1 if client_id in malicious_set else 0 for client_id in range(num_clients)]
    y_pred = [1 if client_id in predicted_set else 0 for client_id in range(num_clients)]
    return float(f1_score(y_true, y_pred, zero_division=0))
