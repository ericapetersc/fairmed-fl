import os
import pickle
from typing import List, Tuple

import hydra
from flwr.common import Metrics, ndarrays_to_parameters
from flwr.server import Server, ServerConfig, SimpleClientManager, start_server
from hydra.core.hydra_config import HydraConfig
from hydra.utils import instantiate
from omegaconf import DictConfig

from src.fairmed_fl.model import get_weights


def weighted_average(metrics: List[Tuple[int, Metrics]]) -> Metrics:
    examples = [num_examples for num_examples, _ in metrics]

    losses = [num_examples * m["loss"] for num_examples, m in metrics]
    auc = [num_examples * m["auc"] for num_examples, m in metrics]
    acc = [num_examples * m["accuracy"] for num_examples, m in metrics]
    min_auc = min([m["auc"] for _, m in metrics])
    min_acc = min([m["accuracy"] for _, m in metrics])

    return {
        "loss": sum(losses) / sum(examples),
        "auc": sum(auc) / sum(examples),
        "accuracy": sum(acc) / sum(examples),
        "min_auc": min_auc,
        "min_accuracy": min_acc,
    }


@hydra.main(config_path=f"../../conf", config_name="federated_base", version_base=None)
def main(cfg: DictConfig) -> None:

    net = instantiate(cfg.model)
    ndarrays = get_weights(net)
    parameters = ndarrays_to_parameters(ndarrays)

    strategy = instantiate(
        cfg.strategy,
        initial_parameters=parameters,
        fit_metrics_aggregation_fn=weighted_average,
        evaluate_metrics_aggregation_fn=weighted_average,
    )
    server = Server(client_manager=SimpleClientManager(), strategy=strategy)
    server.set_max_workers(cfg.server.max_workers)
    config = ServerConfig(cfg.server.num_rounds)

    history = start_server(
        server_address="0.0.0.0:8080",
        server=server,
        config=config,
        # strategy=strategy,
    )

    print(history)

    output_dir = HydraConfig.get().runtime.output_dir
    print(output_dir)

    with open(os.path.join(output_dir, "history.pkl"), "wb") as f_ptr:
        pickle.dump(history, f_ptr)


if __name__ == "__main__":
    main()
