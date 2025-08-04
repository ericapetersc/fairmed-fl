import logging
from collections import defaultdict
from typing import OrderedDict

import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score
from torch import optim
from torchvision.models import (
    DenseNet121_Weights,
    ResNet50_Weights,
    VGG16_BN_Weights,
    densenet121,
    resnet50,
    vgg16_bn,
)
from tqdm import tqdm

from src.fairmed_fl.utils import get_metrics_per_group, save_roc_curve_plot


class CNN(nn.Module):
    def __init__(self, num_classes, pretrained=True, colormode="rgb"):
        super(CNN, self).__init__()
        self.num_classes = num_classes
        self.pretrained = pretrained
        self.colormode = colormode

    def forward(self, x):
        return self.model(x)


class ResNet(CNN):
    def __init__(self, num_classes, pretrained, colormode):
        super(ResNet, self).__init__(num_classes, pretrained, colormode)
        weights = ResNet50_Weights.IMAGENET1K_V1 if self.pretrained else None
        self.model = resnet50(weights=weights)
        if self.colormode == "greyscale":
            self.model.conv1 = nn.Conv2d(
                1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False
            )
        self.model.fc = nn.Linear(self.model.fc.in_features, self.num_classes)


class VGG(CNN):
    def __init__(self, num_classes, pretrained, colormode):
        super(VGG, self).__init__(num_classes, pretrained, colormode)
        weights = VGG16_BN_Weights.IMAGENET1K_V1 if self.pretrained else None
        self.model = vgg16_bn(weights=weights)
        if self.colormode == "greyscale":
            self.model.features[0] = nn.Conv2d(
                1, 64, kernel_size=(3, 3), padding=(1, 1), bias=False
            )
        self.model.classifier[-1] = nn.Linear(
            self.model.classifier[-1].in_features, self.num_classes
        )


class DenseNet(CNN):
    def __init__(self, num_classes, pretrained, colormode):
        super(DenseNet, self).__init__(num_classes, pretrained, colormode)
        weights = DenseNet121_Weights.IMAGENET1K_V1 if self.pretrained else None
        self.model = densenet121(weights=weights)
        if self.colormode == "greyscale":
            self.model.features[0] = nn.Conv2d(
                1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False
            )
        self.model.classifier = nn.Linear(
            self.model.classifier.in_features, self.num_classes
        )


def train_centralized(
    net,
    trainloader,
    valloader,
    device,
    num_epochs,
    steps,
    opt,
    learning_rate,
    momentum,
    weight_decay,
    reduce_lr,
    scheduler_mode,
    scheduler_factor,
    scheduler_patience,
    early_stop,
    early_stop_patience,
    pos_weight=False,
    class_weights=None,
):
    if pos_weight:
        class_weights = class_weights.to(device)

    criterion = nn.BCEWithLogitsLoss(pos_weight=class_weights).to(device)

    if opt == "adam":
        optimizer = optim.Adam(net.parameters(), lr=learning_rate)
    elif opt == "sgd":
        optimizer = optim.SGD(
            net.parameters(),
            lr=learning_rate,
            momentum=momentum,
            weight_decay=weight_decay,
        )

    if reduce_lr:
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer=optimizer,
            mode=scheduler_mode,
            factor=scheduler_factor,
            patience=scheduler_patience,
        )

    best_val_loss = float("inf")
    early_stop_counter = 0

    results = {}

    net.train()
    for epoch in range(num_epochs):
        net.train()
        running_loss = 0.0

        for i, (images, labels, _) in enumerate(trainloader):
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = net(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            if (i + 1) % steps == 0:
                print(
                    f"Epoch {epoch+1}/{num_epochs}, Step {i+1}, Loss: {running_loss / steps:.4f}"
                )
                running_loss = 0.0

        val_results = test(net, valloader, device, class_weights=class_weights)

        results.update({epoch: val_results})

        print(
            f"Epoch {epoch+1}/{num_epochs}, Validation Loss: {val_results['loss']:.4f}"
        )

        if reduce_lr:
            scheduler.step(val_results["loss"])

        if early_stop:
            if val_results["loss"] < best_val_loss:
                best_val_loss = val_results["loss"]
                early_stop_counter = 0
            else:
                early_stop_counter += 1

                if early_stop_counter >= early_stop_patience:
                    break

    return results


def train_fedavg(
    net,
    trainloader,
    num_epochs,
    learning_rate,
    momentum,
    weight_decay,
    device,
    pos_weight=False,
    class_weights=None,
):
    logging.info("Starting training...")

    net.to(device)

    if pos_weight:
        class_weights = class_weights.to(device)

    criterion = nn.BCEWithLogitsLoss(pos_weight=class_weights).to(device)

    optimizer = torch.optim.SGD(
        net.parameters(), lr=learning_rate, momentum=momentum, weight_decay=weight_decay
    )

    net.train()
    for _ in range(num_epochs):
        logging.info("Iterating over dataloader...")
        for images, labels, _ in tqdm(trainloader):
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            loss = criterion(net(images), labels)
            loss.backward()
            optimizer.step()

    results = test(net, trainloader, device, class_weights)

    return results


def test(
    net,
    testloader,
    device,
    class_weights=None,
    return_all_metrics=False,
    classes=None,
    sensitive_attr=None,
    output_dir=None,
    client_id=None,
):
    all_labels = []
    all_outputs = []
    all_attr = defaultdict(list)
    total_loss = 0.0

    criterion = nn.BCEWithLogitsLoss(pos_weight=class_weights).to(device)

    net.eval()
    with torch.no_grad():
        for images, labels, attr in testloader:
            images, labels = images.to(device), labels.to(device)
            outputs = net(images)
            loss = criterion(outputs, labels)
            total_loss += loss.item() * labels.size(0)

            all_outputs.append(outputs.cpu())
            all_labels.append(labels.cpu())
            for key, values in attr.items():
                all_attr[key].extend(values)

    y_logits = torch.cat(all_outputs)
    y_probs = torch.sigmoid(y_logits).numpy()
    y_trues = torch.cat(all_labels).numpy()
    y_preds = (y_probs > 0.5).astype(int)

    try:
        auc = roc_auc_score(y_trues, y_probs, average="macro", multi_class="ovr")
    except:
        auc = None
        logging.info("Ignoring AUC score.")

    results = {
        "loss": total_loss / len(testloader.dataset),
        "auc": auc,
        "accuracy": accuracy_score(y_trues, y_preds),
    }

    if return_all_metrics:

        # Overall metrics
        overall_metrics = results
        overall_metrics.update(
            {
                "precision": precision_score(
                    y_trues, y_preds, average="macro", zero_division=0
                ),
                "recall": recall_score(
                    y_trues, y_preds, average="macro", zero_division=0
                ),
            }
        )

        # Class metrics
        try:
            auc = roc_auc_score(
                y_trues, y_probs, average=None, multi_class="ovr"
            ).tolist()
        except:
            auc = None

        class_metrics = {
            "class": classes,
            "auc": auc,
            "precision": precision_score(
                y_trues, y_preds, average=None, zero_division=0
            ).tolist(),
            "recall": recall_score(
                y_trues, y_preds, average=None, zero_division=0
            ).tolist(),
            "accuracy": [
                accuracy_score(y_trues[:, i], y_preds[:, i])
                for i in range(y_trues.shape[1])
            ],
            "actual_percentage": y_trues.mean(axis=0).tolist(),
            "pred_percentage": y_preds.mean(axis=0).tolist(),
        }

        results = {}
        results.update(
            {
                "overall_metrics": overall_metrics,
                "class_metrics": class_metrics,
            }
        )

        save_roc_curve_plot(output_dir, client_id, classes, y_trues, y_probs)

        results["group_metrics"] = get_metrics_per_group(
            classes, sensitive_attr, y_trues, y_probs, y_preds, all_attr
        )

    return results


def get_weights(net):
    return [val.cpu().numpy() for _, val in net.state_dict().items()]


def set_weights(net, parameters):
    params_dict = zip(net.state_dict().keys(), parameters)
    state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
    net.load_state_dict(state_dict, strict=True)


def get_class_weights(dataloader, num_classes, alpha=None):
    class_counts = torch.zeros(num_classes)
    total_samples = 0

    for _, labels, _ in dataloader:
        class_counts += labels.sum(dim=0)
        total_samples += labels.shape[0]

    class_counts[class_counts == 0] = 1
    class_weights = total_samples / class_counts

    if alpha:
        return class_weights * alpha

    return class_weights
