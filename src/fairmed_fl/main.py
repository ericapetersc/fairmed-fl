import json
import logging

import hydra
import torch
from hydra.core.hydra_config import HydraConfig
from hydra.utils import instantiate
from omegaconf import DictConfig
from torch.utils.data import DataLoader

from src.fairmed_fl.model import get_class_weights, test, train_centralized
from src.fairmed_fl.utils import save_json_results


@hydra.main(
    config_path=f"../../conf", config_name="centralized_base", version_base=None
)
def main(cfg: DictConfig) -> None:

    output_dir = HydraConfig.get().runtime.output_dir

    partitioner = instantiate(cfg.partitioner)

    classes = partitioner.classes
    sensitive_attr = partitioner.sensitive_attr_cols

    for cid in range(partitioner.num_clients):

        logging.info(f"Starting client {cid} train...")

        trainset, valset, testset = partitioner.load_client_datasets(cid, cfg.dataset)

        trainloader = DataLoader(
            trainset,
            batch_size=cfg.batch_size,
            shuffle=True,
            num_workers=cfg.num_workers,
        )
        valloader = DataLoader(
            valset, batch_size=cfg.batch_size, shuffle=True, num_workers=cfg.num_workers
        )
        testloader = DataLoader(
            testset,
            batch_size=cfg.batch_size,
            shuffle=False,
            num_workers=cfg.num_workers,
        )

        device = torch.device(
            "cuda" if torch.cuda.is_available() and cfg.use_cuda else "cpu"
        )

        net = instantiate(cfg.model).to(device)

        class_weights = None
        if cfg.pos_weight:
            class_weights = get_class_weights(trainloader, len(classes), cfg.alpha)

        train_results = train_centralized(
            net,
            trainloader,
            valloader,
            device,
            num_epochs=cfg.num_epochs,
            steps=cfg.steps,
            opt=cfg.opt,
            learning_rate=cfg.learning_rate,
            momentum=cfg.momentum,
            weight_decay=cfg.weight_decay,
            reduce_lr=cfg.reduce_lr,
            scheduler_mode=cfg.scheduler_mode,
            scheduler_factor=cfg.scheduler_factor,
            scheduler_patience=cfg.scheduler_patience,
            early_stop=cfg.early_stop,
            early_stop_patience=cfg.early_stop_patience,
            pos_weight=cfg.pos_weight,
            class_weights=class_weights,
        )

        save_json_results(output_dir, f"client_{cid}_training.json", train_results)
        logging.info(f"Client {cid} training steps save on {cid}_training.json.")

        logging.info(f"Starting client {cid} test...")

        results = test(
            net,
            testloader,
            device,
            class_weights,
            True,
            classes,
            sensitive_attr,
            output_dir,
            cid,
        )

        save_json_results(output_dir, f"client_{cid}_metrics.json", results)
        logging.info(f"Client {cid} metrics save on {cid}_metrics.json.")


if __name__ == "__main__":
    main()
