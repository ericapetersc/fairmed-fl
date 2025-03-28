import logging
import os
from typing import Optional, OrderedDict, Union

import flwr as fl
import numpy as np
import torch
from flwr.common import FitRes, Parameters, Scalar
from flwr.server.client_proxy import ClientProxy


class CustomFedAvg(fl.server.strategy.FedAvg):
    def __init__(self, output_dir, num_rounds, save_steps, net, device, **kwargs):
        super().__init__(**kwargs)
        self.output_dir = output_dir
        self.num_rounds = num_rounds
        self.save_steps = save_steps
        self.net = net
        self.device = device

    def aggregate_fit(
        self,
        server_round: int,
        results: list[tuple[fl.server.client_proxy.ClientProxy, fl.common.FitRes]],
        failures: list[Union[tuple[ClientProxy, FitRes], BaseException]],
    ) -> tuple[Optional[Parameters], dict[str, Scalar]]:

        aggregated_parameters, aggregated_metrics = super().aggregate_fit(
            server_round, results, failures
        )

        if aggregated_parameters is not None:
            aggregated_ndarrays: list[np.ndarray] = fl.common.parameters_to_ndarrays(
                aggregated_parameters
            )

            if server_round % self.save_steps == 0 or server_round == self.num_rounds:
                self.net = self.net.to(self.device)
                params_dict = zip(self.net.state_dict().keys(), aggregated_ndarrays)
                state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
                self.net.load_state_dict(state_dict, strict=True)

                logging.info(f"Saving round {server_round} model weights...")
                torch.save(
                    self.net.state_dict(),
                    os.path.join(self.output_dir, f"model_round_{server_round}.pth"),
                )

        return aggregated_parameters, aggregated_metrics
