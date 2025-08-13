import logging
import math
from dataclasses import dataclass
from typing import List, Optional

import numpy as np
import pandas as pd
import torch
from imblearn.over_sampling import RandomOverSampler
from PIL import Image
from torchvision.transforms import (
    CenterCrop,
    Compose,
    Normalize,
    RandomHorizontalFlip,
    RandomRotation,
    Resize,
    ToTensor,
)

TRANSFORMER = Compose(
    [
        Resize((320, 320)),
        CenterCrop(224),
        RandomHorizontalFlip(),
        RandomRotation(10),
        ToTensor(),
        Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)


@dataclass
class DatasetConfig:
    data_path: str
    patient_id_col: str
    num_classes: int
    classes: List[str]
    colormode: str = "rgb"
    transform: Optional[Compose] = TRANSFORMER
    policy: Optional[str] = None
    sensitive_attr: Optional[List[str]] = None


class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, data, config: DatasetConfig, balance_data: bool = False):
        self.patient_id_col = config.patient_id_col
        self.num_classes = config.num_classes
        self.classes = list(config.classes)
        self.colormode = config.colormode
        self.transform = config.transform
        self.policy = config.policy
        self.sensitive_attr = list(config.sensitive_attr)
        self.balance_data = balance_data
        self.data = self._preprocess_data(data)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):

        item = self.data.iloc[idx]
        img_path = item["image"]

        if self.colormode == "greyscale":
            image = Image.open(img_path).convert("L")
        else:
            image = Image.open(img_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        label = torch.tensor(item[self.classes].to_list(), dtype=torch.float32)

        if self.sensitive_attr:
            attrib = item[self.sensitive_attr].to_dict()

        return image, label, attrib

    def _preprocess_data(self, data):
        if self.policy:
            data = self._apply_policy(data)
        if self.sensitive_attr:
            data = self._filter_rows_with_nan_attrib(data)
        if self.balance_data:
            data = self._balance_dataset(data)
        return data

    def _apply_policy(self, data):
        if self.policy == "ones":
            value = 1
        elif self.policy == "zeros":
            value = 0
        else:
            raise ValueError(
                "Invalid policy '{policy}'. Choose between 'zeros' or 'ones'."
            )
        data.loc[:, self.classes] = data.loc[:, self.classes].replace(-1, value)
        return data

    def _filter_rows_with_nan_attrib(self, data):
        data = data[data[self.sensitive_attr].notna().apply(lambda x: all(x), axis=1)]
        return data.reset_index(drop=True)

    def _balance_dataset(self, data):
        if len(self.sensitive_attr) > 1:
            raise ValueError(
                "Dataset balance only possible for one sensitive attribute."
            )

        logging.info("Resampling data...")

        values = data[self.sensitive_attr[0]].value_counts().to_dict()
        mean = np.mean(list(values.values()))

        max_samples = max(list(values.values()))

        new_sample_sizes = {}

        for k, v in values.items():
            if v < mean:
                factor = math.sqrt(max_samples / v)
                new_sample_sizes[k] = int(factor * v)

            ros = RandomOverSampler(sampling_strategy=new_sample_sizes)

            y = data[self.sensitive_attr[0]]
            X = data.drop(columns=self.sensitive_attr[0])

            X_resampled, y_resampled = ros.fit_resample(X, y)

        resampled_data = pd.concat(
            [X_resampled, y_resampled], axis=1, ignore_index=True
        )
        resampled_data.columns = list(X.columns) + self.sensitive_attr

        logging.info(
            f"Orignal dataset had {data.shape[0]}. New dataset has {resampled_data.shape[0]}"
        )

        return resampled_data
