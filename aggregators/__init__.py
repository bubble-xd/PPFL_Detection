from .clustering import clustering_aggregate
from .common import AggregationResult
from .fedavg import fedavg_aggregate
from .krum import krum_aggregate, multi_krum_aggregate
from .median import median_aggregate


def _normalize_method_name(method_name: str) -> str:
    normalized = str(method_name).strip().lower().replace("-", "_")
    aliases = {
        "muti_krum": "multi_krum",
    }
    return aliases.get(normalized, normalized)


def run_aggregator(
    method_name: str,
    local_state_dicts,
    feature_matrix,
    num_malicious: int,
    global_state_dict,
    config,
) -> AggregationResult:
    normalized_name = _normalize_method_name(method_name)
    if normalized_name == "fedavg":
        return fedavg_aggregate(local_state_dicts, global_state_dict=global_state_dict)
    if normalized_name == "krum":
        return krum_aggregate(
            local_state_dicts,
            feature_matrix,
            config.get_krum_f(default_num_malicious=num_malicious),
            global_state_dict=global_state_dict,
        )
    if normalized_name == "multi_krum":
        return multi_krum_aggregate(
            local_state_dicts,
            feature_matrix,
            config.get_multi_krum_f(default_num_malicious=num_malicious),
            m=config.get_multi_krum_m(default_num_malicious=num_malicious),
            global_state_dict=global_state_dict,
        )
    if normalized_name == "median":
        return median_aggregate(
            local_state_dicts,
            feature_matrix,
            global_state_dict=global_state_dict,
            max_iters=config.GEOM_MEDIAN_MAX_ITERS,
            tol=config.GEOM_MEDIAN_TOL,
            num_malicious=num_malicious,
        )
    if normalized_name == "clustering":
        return clustering_aggregate(
            local_state_dicts,
            feature_matrix,
            global_state_dict=global_state_dict,
            random_state=config.KMEANS_RANDOM_STATE,
            n_init=config.KMEANS_N_INIT,
        )
    raise ValueError(f"Unsupported aggregation method: {method_name}")


__all__ = ["AggregationResult", "run_aggregator"]
