import logging
from typing import Callable

import hydra
import torch
from flwr.client import NumPyClient, start_client
from hydra.core.hydra_config import HydraConfig
from hydra.utils import instantiate
from omegaconf import DictConfig
from torch.utils.data import DataLoader

from src.fairmed_fl.model import (
    get_class_weights,
    get_weights,
    set_weights,
    test,
    train_fedavg,
)
from src.fairmed_fl.utils import save_json_results


class Client(NumPyClient):
    def __init__(
        self,
        client_id,
        net,
        trainset,
        valset,
        testset,
        batch_size,
        num_workers,
        device,
        num_epochs,
        learning_rate,
        momentum,
        weight_decay,
        pos_weight,
        alpha,
        output_dir,
        classes,
        sensitive_attr,
    ):
        self.client_id = client_id
        self.trainset = trainset
        self.valset = valset
        self.testset = testset
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.device = device
        self.num_epochs = num_epochs
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.weight_decay = weight_decay
        self.pos_weight = pos_weight
        self.alpha = alpha
        self.output_dir = output_dir
        self.classes = classes
        self.sensitive_attr = sensitive_attr

        self.net = net.to(self.device)

        self.trainloader, self.valloader, self.testloader = self.get_dataloaders()

        self.class_weights = None
        if self.pos_weight:
            self.class_weights = get_class_weights(
                self.trainloader, len(classes), self.alpha
            )

    def get_dataloaders(self):
        return (
            DataLoader(
                self.trainset,
                batch_size=self.batch_size,
                shuffle=True,
                num_workers=self.num_workers,
            ),
            DataLoader(
                self.valset,
                batch_size=self.batch_size,
                shuffle=True,
                num_workers=self.num_workers,
            ),
            DataLoader(
                self.testset,
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=self.num_workers,
            ),
        )

    def fit(self, parameters, config):
        set_weights(self.net, parameters)
        results = train_fedavg(
            self.net,
            self.trainloader,
            num_epochs=self.num_epochs,
            learning_rate=self.learning_rate,
            momentum=self.momentum,
            weight_decay=self.weight_decay,
            device=self.device,
            pos_weight=self.pos_weight,
            class_weights=self.class_weights,
        )
        return get_weights(self.net), len(self.trainloader.dataset), results

    def evaluate(self, parameters, config):
        set_weights(self.net, parameters)
        results = test(self.net, self.valloader, self.device, self.class_weights)
        return results["loss"], len(self.valloader.dataset), results

    def save_test_results(self):
        results = test(
            self.net,
            self.testloader,
            self.device,
            self.class_weights,
            True,
            self.classes,
            self.sensitive_attr,
            self.output_dir,
            self.client_id,
        )
        save_json_results(
            self.output_dir, f"client_{self.client_id}_metrics.json", results
        )


def get_client_fn(cfg, output_dir) -> Callable[[str], Client]:

    def client_fn(cid: str):

        partitioner = instantiate(cfg.partitioner)
        trainset, valset, testset = partitioner.load_client_datasets(
            int(cid), cfg.dataset
        )

        logging.info("Dataset {cid} loaded.")

        net = instantiate(cfg.model)

        device = torch.device(
            "cuda" if torch.cuda.is_available() and cfg.client.use_cuda else "cpu"
        )

        return Client(
            int(cid),
            net,
            trainset,
            valset,
            testset,
            cfg.client.batch_size,
            cfg.client.num_workers,
            device,
            cfg.client.num_epochs,
            cfg.client.learning_rate,
            cfg.client.momentum,
            cfg.client.weight_decay,
            cfg.client.pos_weight,
            cfg.client.alpha,
            output_dir,
            cfg.dataset.config.classes,
            cfg.dataset.config.sensitive_attr,
        )

    return client_fn


@hydra.main(config_path=f"../../conf", config_name="federated_base", version_base=None)
def main(cfg: DictConfig) -> None:

    output_dir = HydraConfig.get().runtime.output_dir

    partitioner = instantiate(cfg.partitioner)
    classes = partitioner.classes
    sensitive_attr = partitioner.sensitive_attr_cols

    trainset, valset, testset = partitioner.load_client_datasets(
        cfg.client.id, cfg.dataset
    )

    net = instantiate(cfg.model)

    device = torch.device(
        "cuda" if torch.cuda.is_available() and cfg.client.use_cuda else "cpu"
    )

    client = Client(
        cfg.client.id,
        net,
        trainset,
        valset,
        testset,
        cfg.client.batch_size,
        cfg.client.num_workers,
        device,
        cfg.client.num_epochs,
        cfg.client.learning_rate,
        cfg.client.momentum,
        cfg.client.weight_decay,
        output_dir,
        classes,
        sensitive_attr,
    )

    logging.info(f"Starting Client {cfg.client.id}...")

    start_client(server_address="127.0.0.1:8080", client=client.to_client())

    client.save_test_results()

    logging.info(f"Client {cfg.client.id} metrics saved in {output_dir} directory.")


if __name__ == "__main__":
    main()
