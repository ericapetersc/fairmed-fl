import glob
import logging
import os
import pickle

import flwr as fl
import hydra
import torch
from flwr.common import Metrics, ndarrays_to_parameters
from flwr.server.client_manager import SimpleClientManager
from flwr.server.server import Server
from hydra.core.hydra_config import HydraConfig
from hydra.utils import call, instantiate
from omegaconf import DictConfig
from torch.utils.data import DataLoader

from src.fairmed_fl.model import get_class_weights, get_weights, test
from src.fairmed_fl.server import weighted_average
from src.fairmed_fl.utils import save_json_results


@hydra.main(config_path=f"../../conf", config_name="federated_base", version_base=None)
def main(cfg: DictConfig) -> None:

    output_dir = HydraConfig.get().runtime.output_dir

    net = instantiate(cfg.model)
    ndarrays = get_weights(net)
    parameters = ndarrays_to_parameters(ndarrays)

    client_fn = call(
        cfg.simulation.client_fn,
        cfg,
        output_dir,
    )

    strategy = instantiate(
        cfg.strategy,
        initial_parameters=parameters,
        fit_metrics_aggregation_fn=weighted_average,
        evaluate_metrics_aggregation_fn=weighted_average,
        output_dir=output_dir,
        num_rounds=cfg.server.num_rounds,
        net=instantiate(cfg.model),
        device=torch.device(
            "cuda" if torch.cuda.is_available() and cfg.client.use_cuda else "cpu"
        ),
    )

    server = Server(strategy=strategy, client_manager=SimpleClientManager())

    history = fl.simulation.start_simulation(
        server=server,
        client_fn=client_fn,
        num_clients=cfg.server.num_clients,
        config=fl.server.ServerConfig(num_rounds=cfg.server.num_rounds),
        client_resources={
            "num_cpus": cfg.simulation.client_resources.num_cpus,
            "num_gpus": cfg.simulation.client_resources.num_gpus,
        },
        strategy=strategy,
    )

    print(history)

    with open(os.path.join(output_dir, "history.pkl"), "wb") as f_ptr:
        pickle.dump(history, f_ptr)

    if cfg.simulation.clients_test:
        partitioner = instantiate(cfg.partitioner)
        classes = partitioner.classes

        for cid in range(partitioner.num_clients):
            logging.info(f"Starting client {cid} test...")
            trainloader, _, testset = partitioner.load_client_datasets(cid, cfg.dataset)

            class_weights = None
            if cfg.client.pos_weight:
                class_weights = get_class_weights(
                    trainloader, len(classes), cfg.client.alpha
                )

            testloader = DataLoader(
                testset,
                batch_size=cfg.client.batch_size,
                shuffle=False,
                num_workers=cfg.client.num_workers,
            )

            device = torch.device(
                "cuda" if torch.cuda.is_available() and cfg.client.use_cuda else "cpu"
            )

            files = [
                fname for fname in glob.glob(os.path.join(output_dir, "model_round_*"))
            ]
            latest_round_file = max(files, key=os.path.getctime)
            logging.info(f"Loading global model from {latest_round_file}")
            state_dict = torch.load(latest_round_file)
            net = instantiate(cfg.model).to(device)
            net.load_state_dict(state_dict)

            results = test(
                net,
                testloader,
                device,
                class_weights,
                True,
                partitioner.classes,
                partitioner.sensitive_attr_cols,
                output_dir,
                cid,
            )

            save_json_results(output_dir, f"client_{cid}_metrics.json", results)

            logging.info(f"Client {cid} metrics save on {cid}_metrics.json.")


if __name__ == "__main__":
    main()
