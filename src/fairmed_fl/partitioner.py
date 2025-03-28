import logging
import os
import random
import warnings
from dataclasses import dataclass
from typing import List, Union

import hydra
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from hydra.core.hydra_config import HydraConfig
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

warnings.simplefilter(action="ignore", category=FutureWarning)


@dataclass
class PartitionConfig:
    classes: List
    num_clients: int
    patient_id_col: str
    sensitive_attr_cols: List[str]
    val_ratio: float
    test_ratio: float
    alpha: float = None  # Used for DirichletPartitioner
    partition_by: List[str] = None  # Used for DirichletPartitioner
    min_samples_to_partition: int = None  # Used for DirichletPartitioner
    datadir: str = None
    plot_clients: bool = True


class Partitioner:
    def __init__(self, config: PartitionConfig):
        self.classes = list(config.classes)
        self.num_clients = config.num_clients
        self.patient_id_col = config.patient_id_col
        self.sensitive_attr_cols = list(config.sensitive_attr_cols)
        self.val_ratio = config.val_ratio
        self.test_ratio = config.test_ratio
        self.datadir = config.datadir
        self.plot_clients = config.plot_clients

    def split_train_test_data(self, data):
        patient_ids = data[self.patient_id_col].unique()

        train_ids, test_ids = train_test_split(patient_ids, test_size=self.test_ratio)
        train_ids, val_ids = train_test_split(
            train_ids, test_size=(self.val_ratio / (1 - self.test_ratio))
        )

        traindata = data[data[self.patient_id_col].isin(train_ids)].copy()
        valdata = data[data[self.patient_id_col].isin(val_ids)].copy()
        testdata = data[data[self.patient_id_col].isin(test_ids)].copy()

        traindata["split"] = "train"
        valdata["split"] = "val"
        testdata["split"] = "test"

        return pd.concat([traindata, valdata, testdata], ignore_index=True)

    def save_partitioned_clients(self, clients):
        if not self.datadir:
            logging.error("Data directory is not specified.")
        os.makedirs(self.datadir, exist_ok=True)
        for i, data in enumerate(clients):
            data.to_csv(os.path.join(self.datadir, f"client_{i}.csv"), index=False)
        logging.info(f"Clients saved in {self.datadir}.")

    def execute(self, data):
        logging.info("Partitioning data...")
        clients = self.partition(data)
        if "split" in data.columns:
            if self.val_ratio and self.test_ratio:
                logging.warning(
                    "'Split' column already exists in data. Ignoring val_ratio and test_ratio..."
                )
        else:
            if self.val_ratio and self.test_ratio:
                logging.info("Spliting data into train, val and test...")
                clients = [self.split_train_test_data(c) for c in clients]
        self.save_partitioned_clients(clients)
        if self.plot_clients:
            self.save_clients_data_distribution_plot(
                clients, self.classes, "classes_distributions"
            )

            self.save_clients_data_distribution_plot(
                clients, self.sensitive_attr_cols, "sensitive_attr_distributions"
            )

    def load_client_datasets(self, cid, config):
        if not self.datadir:
            logging.error("Data directory is not specified.")
        data = pd.read_csv(os.path.join(self.datadir, f"client_{cid}.csv"))
        return (
            instantiate(config, data=data[data["split"] == "train"]),
            instantiate(config, data=data[data["split"] == "val"]),
            instantiate(config, data=data[data["split"] == "test"]),
        )

    def save_clients_data_distribution_plot(self, clients, columns, figname):
        num_columns = len(columns)
        fig_cols = 2
        fig_rows = (num_columns + 1) // 2

        fig, axes = plt.subplots(
            fig_rows, fig_cols, figsize=(12, fig_rows * 5), constrained_layout=True
        )
        axes = axes.flatten()

        for i, col_name in enumerate(columns):
            ax = axes[i]
            col_distributions = []
            col_values = set()

            for client in clients:
                col_counts = client[col_name].value_counts(normalize=True)
                col_values.update(col_counts.index)
                col_distributions.append(col_counts)

            col_values = sorted(col_values)
            col_matrix = np.zeros((len(clients), len(col_values)))

            for col_id, distribution in enumerate(col_distributions):
                for val_id, col_val in enumerate(col_values):
                    col_matrix[col_id, val_id] = distribution.get(col_val, 0)

            bottom = np.zeros(len(clients))

            for cid, col_val in enumerate(col_values):
                ax.bar(
                    range(len(clients)),
                    col_matrix[:, cid],
                    bottom=bottom,
                    label=str(col_val),
                )
                bottom += col_matrix[:, cid]

            ax.set_title(f"{col_name}: Distribution Across Clients", fontsize=12)
            ax.set_xlabel("Clients", fontsize=10)
            ax.set_ylabel("Proportion", fontsize=10)
            ax.set_xticks(range(len(clients)))
            ax.legend(
                title="Values", fontsize=8, bbox_to_anchor=(1.05, 1), loc="upper left"
            )

        for j in range(i + 1, len(axes)):
            if j == len(axes) - 1 and num_columns % 2 != 0:
                axes[j].axis("off")
                axes[j].set_position([0.25, 0.1, 0.5, 0.5])
            else:
                axes[j].axis("off")

        os.makedirs(self.datadir, exist_ok=True)
        fig.savefig(os.path.join(self.datadir, f"{figname}.png"))
        logging.info(f"Client's {figname} plot saved in {self.datadir}.")


class IidPartitioner(Partitioner):
    def __init__(self, config: PartitionConfig):
        super().__init__(config)

    def partition(self, data):
        patient_ids = data[self.patient_id_col].unique()
        random.shuffle(patient_ids)
        clients = np.array_split(patient_ids, self.num_clients)
        return [
            pd.concat(
                [data[data[self.patient_id_col] == pid] for pid in client]
            ).reset_index(drop=True)
            for client in clients
        ]


class DirichletPartitioner(Partitioner):
    def __init__(self, config: PartitionConfig):
        super().__init__(config)
        self.alpha = config.alpha
        self.partition_by = list(config.partition_by)
        self.min_samples_to_partition = config.min_samples_to_partition
        self.categories = None
        self.num_categories = None
        self.proportions = None

    def aggregate_attributes(self, group):
        """
        Necessary to work on multiple sampes from the same patient_id. Each patient is represented by the most frequent value of each sensitive attribute.
        """
        agg_data = {}
        for col in self.partition_by:
            agg_data[col] = group[col].mode()[0]  # Mode for most frequent attribute
        return pd.Series(agg_data)

    def calculate_proportions(self):
        """
        Get an array with each category proportion for each client. The proportions of each class across clients sums equal to 1.
        """
        proportions = np.zeros((self.num_clients, self.num_categories))
        for catid in range(self.num_categories):
            proportions[:, catid] = np.random.dirichlet([self.alpha] * self.num_clients)
        self.proportions = proportions

    def partition(self, data):
        if any([any(data[attr].isna()) for attr in self.partition_by]):
            raise Exception("Invalid value for sensitive attribute.")

        data = data.copy()

        clients = [pd.DataFrame(columns=data.columns) for _ in range(self.num_clients)]

        if self.min_samples_to_partition:
            small_groups = dict()
            for attr in self.partition_by:
                small_groups.update(
                    {
                        attr: data[
                            data[attr].map(data[attr].value_counts())
                            < self.min_samples_to_partition
                        ][attr].unique()
                    }
                )

            cid = 0
            ids_to_exclude = []

            for attr, categories in small_groups.items():
                if categories.size > 0:
                    for cat in categories:
                        cat_patient_ids = data[data[attr] == cat][
                            self.patient_id_col
                        ].unique()
                        cat_data = data[data[self.patient_id_col].isin(cat_patient_ids)]
                        clients[cid] = pd.concat([clients[cid], cat_data], axis=0)
                        cid = (cid + 1) % self.num_clients

                        ids_to_exclude = np.concatenate(
                            [ids_to_exclude, cat_patient_ids]
                        )

            data = data[~data.patient_id.isin(ids_to_exclude)]

        unique_patients = (
            data.groupby(self.patient_id_col)[self.partition_by]
            .apply(self.aggregate_attributes)
            .reset_index()
        )

        if len(self.partition_by) > 1:
            attr_col = (
                unique_patients[self.partition_by].astype(str).agg("_".join, axis=1)
            )
        else:
            attr_col = unique_patients[self.partition_by[0]]

        le = LabelEncoder()
        unique_patients["label_encoded"] = le.fit_transform(attr_col)

        self.categories = le.classes_.tolist()
        self.num_categories = len(self.categories)

        self.calculate_proportions()

        for catid in range(self.num_categories):
            cat_patient_ids = unique_patients[
                unique_patients["label_encoded"] == catid
            ][self.patient_id_col].values
            np.random.shuffle(cat_patient_ids)

            splits = np.cumsum(
                np.floor(self.proportions[:, catid] * len(cat_patient_ids)).astype(int)
            )[:-1]
            partitions = np.split(cat_patient_ids, splits)

            for cid, partition in enumerate(partitions):
                partition_cat_data = data[data[self.patient_id_col].isin(partition)]
                clients[cid] = pd.concat([clients[cid], partition_cat_data], axis=0)

        return [client.reset_index(drop=True) for client in clients]


@hydra.main(
    config_path=f"../../conf", config_name="centralized_base", version_base=None
)
def main(cfg: DictConfig) -> None:

    data = pd.read_csv(cfg.dataset.config.data_path)
    partitioner = instantiate(cfg.partitioner)
    partitioner.execute(data)


if __name__ == "__main__":
    main()
