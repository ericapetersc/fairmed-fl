# FairMed-FL: Federated Learning for Fair and Unbiased Deep Learning in Medical Imaging

This repository contains the source code for FairMed-FL, a framework for applying federated learning to medical imaging with a focus on group fairness.

**Authors:** Êrica Peters do Carmo, Caetano Traina Junior, and Agma J. M. Traina

## Step-by-Step for Running FairMED-FL

### Setup

1. Clone the repository:

```bash
git clone https://github.com/ericapetersc/fairmed-fl.git
cd fairmed-fl
```

2. Create and activate a virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

### Add Datasets

The published work utilized two publicly available chest X-ray datasets for disease classification: **NIH ChestX-ray8** and **CheXpert**. Dataset configuration files are located in:

```
fairmed-fl/
└───conf/
    └───datasets/
        ├── chexpert.yaml
        └── nih.yaml
```

To add a new dataset, create a .yaml file in the same directory with the following structure:

```yaml
_target_: src.fairmed_fl.dataset.CustomDataset
config:
  _target_: src.fairmed_fl.dataset.DatasetConfig
  data_path: {/your/data/path.csv}
  patient_id_col: {'patient_id'} # Patient identifier
  num_classes: {number_of_classes}
  classes:
    - 'class1'
    - 'class2'
    - 'class3'
  policy: {'ones' | 'zeros'}  # Determines label handling strategy
  sensitive_attr:
    - 'attr1'  # e.g., 'sex', 'age'
    - 'attr2'
```

### Partitioning the Data

Once you’ve chosen a dataset, you can partition it across N clients using one of two available strategies: IID or Dirichlet partitioning.

Dataset partitioning configurations are located in:

```
fairmed-fl/
└───conf/
    └───partitioner/
        ├── dirichlet.yaml
        └── iid.yaml
```

Each partitioner is configured using the following structure:

```yaml
_target_: src.fairmed_fl.partitioner.IidPartitioner | src.fairmed_fl.partitioner.DirichletPartitioner
config:
  _target_: src.fairmed_fl.partitioner.PartitionConfig
  classes: ${dataset.config.classes}
  num_clients: 5  # Number of clients to split the dataset into
  patient_id_col: ${dataset.config.patient_id_col}  # Ensures each patient's data appears in only one client and one data split (train/val/test)
  sensitive_attr_cols: ${dataset.config.sensitive_attr}
  val_ratio: 0.1  # Validation set ratio
  test_ratio: 0.2  # Test set ratio
  alpha: 0.5  # Only used for Dirichlet partitioning
  partition_by: ${dataset.config.sensitive_attr}  # Sensitive attributes used to guide Dirichlet partitioning
  min_samples_to_partition: 1000  # Minimum number of samples required for splitting. If a sensitive attribute group has fewer samples, it is assigned to a single client.
  datadir: ./clients  # Output folder for the client CSV files
  plot_clients: True  # Whether to generate and save distribution plots of classes and sensitive attributes
```

Partitioning is controlled by the conf/centralized_based.yaml configuration file. Make sure it references the correct dataset and partitioner in the defaults section:

```yaml
defaults:
  - _self_
  - dataset: {your_dataset}
  - partitioner: {iid | dirichlet}
```

To run the partitioning process, execute the following command:

```bash
make run-partitioning
```

The resulting client datasets (as CSV files) and distribution plots will be saved in the directory specified by datadir (default is ./clients).

### Run Centralized Training

To train a centralized model for each client, ensure that all client CSV files are located in the folder specified in the partitioner YAML file.

Then, run the following command in the terminal:


```bash
make run-centralized
```

You can configure training parameters in your Hydra config file. A typical configuration looks like this:

```yaml
defaults:
  - _self_
  - dataset: {your-dataset}
  - partitioner: {iid | dirichlet}

batch_size: 32
num_workers: 4
use_cuda: True
num_epochs: 2
steps: 100  # Number of steps between validation checks

# Optimizer settings
opt: {adam | sgd}
learning_rate: 0.01
momentum: 0.9        # Only used when opt = sgd
weight_decay: 0.0001 # Only used when opt = sgd

# Learning rate scheduler
reduce_lr: true                 # Whether to reduce LR on plateau
scheduler_mode: 'min'           # 'min' or 'max' for monitoring metric
scheduler_factor: 0.1           # Factor by which the LR will be reduced
scheduler_patience: 5           # Number of epochs with no improvement before reducing LR

# Early stopping
early_stop: true
early_stop_patience: 10         # Number of epochs with no improvement before stopping

# Loss weighting
pos_weight: false               # Whether to apply positive class weighting in BCE loss
alpha: 0.2                      # Scaling factor for positive weights (used if pos_weight = true)

# Model configuration
model:
  _target_: src.fairmed_fl.model.DenseNet
  num_classes: ${dataset.config.num_classes}
  pretrained: true
  colormode: "rgb"
```

---
**NOTE:**

* The current implementation supports **multi-label classification only**.
* The default loss function is **Binary Cross Entropy (BCE)**.
---

### Run Federated Training

To run federated training, ensure all client CSV files are located in the folder specified in the partitioner YAML file. Then, execute the following command in the terminal:


```bash
make run-simulation
```

---
**NOTE:**

* Currently, the only supported federated strategy is Federated Averaging (FedAvg) using the SGD optimizer.
---

You can customize training and simulation behavior in your Hydra configuration. A typical setup looks like this:

```yaml
defaults:
  - _self_
  - dataset: {your-dataset}
  - partitioner: {iid | dirichlet}
  - strategy: fedavg

server:
  num_clients: ${partitioner.config.num_clients}  # Total number of clients
  max_workers: 4
  num_rounds: 100  # Number of federated communication rounds

model:
  _target_: src.fairmed_fl.model.DenseNet
  num_classes: ${dataset.config.num_classes}
  pretrained: true
  colormode: "rgb"

client:
  id: 0
  batch_size: 32
  num_workers: 4
  use_cuda: true
  num_epochs: 3        # Number of local epochs per round
  learning_rate: 0.01
  momentum: 0.9
  weight_decay: 0.0001
  pos_weight: false
  alpha: 0.2            # Scaling factor for positive class weights (used if pos_weight = true)

simulation:
  client_fn:
    _target_: src.fairmed_fl.client.get_client_fn
  client_resources:
    num_cpus: 16
    num_gpus: 0.25
  clients_test: true     # Whether to test each client individually using the global model
```

### Interpret Results

**Centralized Training**

In centralized training, a separate model is trained for each client using only that client's data. Each model is then evaluated on the corresponding client’s test set.

**Federated Learning**

In federated training, a single global model is collaboratively trained using all clients' data (in a privacy-preserving way). This global model is then evaluated on each client’s individual test set.

**Outputs Files**

After training—regardless of centralized or federated mode—the following result files are generated for each client and saved in the outputs/ folder:

```
fairmed-fl
└───outputs
│      ├── client_{n}_plot.png         # ROC-AUC curve for all classes (Client n)
│      ├── client_{n}_training.json    # Validation metrics during training (Client n)
│      └── client_{n}_metrics.json     # Final test metrics after training (Client n)
```

**Training Metrics**

The training log files (client_{n}_training.json) follow this structure:

```json
{
    "epoch": {
        "loss": float,
        "auc": float,
        "accuracy": float
    }
}
```

**Test Metrics**

The test metrics file {client_{n}_metrics.json} contains the evaluation results of the model on the client’s test set. It is divided into three main sections: overall metrics, class metrics and group metrics.

```json
{
    "overall_metrics": {
        "loss": float,
        "auc": float,
        "accuracy": float,
        "precision": float,
        "recall": float
    },
    "class_metrics": {
        "class": ["class1", "class2", ..., "classN"],
        "auc": [float, float, ..., float],
        "precision": [float, float, ..., float],
        "recall": [float, float, ..., float],
        "accuracy": [float, float, ..., float],
        "actual_percentage": [float, float, ..., float],  // True prevalence in dataset
        "pred_percentage": [float, float, ..., float]     // Model’s predicted prevalence
    },
    "group_metrics": {
        "sensitive_attr1": {
            "value1": {
                "overall_metrics": {
                    "auc": float,
                    "accuracy": float,
                    "precision": float,
                    "recall": float,
                    "num_samples": int,
                    "mean_tpr": float,
                    "mean_fpr": float,
                    "mean_fnr": float,
                    "mean_tnr": float,
                    "mean_ppr": float
                },
                "class_metrics": [
                    {
                        "class_name": "class1",
                        "tpr": float,
                        "fpr": float,
                        "fnr": float,
                        "tnr": float,
                        "ppr": float,
                        "class_distribution": float
                    },
                    {
                        "class_name": "class2",
                        ...
                    }
                ]
            },
            "value2": {
                ...
            }
        }
    }
}
```

## License

The FairMed-FL is released under the MIT License.