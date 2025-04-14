import json
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    auc,
    confusion_matrix,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)


def save_roc_curve_plot(
    output_dir, client_id, classes, y_trues, y_probs, figsize=(9, 9)
):
    fig, ax = plt.subplots(1, 1, figsize=figsize)
    for idx, class_name in enumerate(classes):
        fpr, tpr, _ = roc_curve(y_trues[:, idx], y_probs[:, idx])
        ax.plot(fpr, tpr, label=f"{class_name} (AUC: {auc(fpr, tpr):0.2f})")

    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curves for Each Class")
    ax.legend()
    plt.savefig(os.path.join(output_dir, f"client_{str(client_id)}_plot.png"))


def get_metrics_per_group(classes, sensitive_attr, y_trues, y_probs, y_preds, all_attr):

    df = pd.DataFrame(
        {
            **{
                f"{class_name}_prob": y_probs[:, i]
                for i, class_name in enumerate(classes)
            },
            **{
                f"{class_name}_pred": y_preds[:, i]
                for i, class_name in enumerate(classes)
            },
            **{
                f"{class_name}_label": y_trues[:, i]
                for i, class_name in enumerate(classes)
            },
            **all_attr,
        }
    )

    results = {}

    for attr in sensitive_attr:
        group_metrics = {}

        for group in df[attr].unique():
            group_df = df[df[attr] == group]

            y_trues_group = group_df[[f"{cls}_label" for cls in classes]].to_numpy()
            y_probs_group = group_df[[f"{cls}_prob" for cls in classes]].to_numpy()
            y_preds_group = group_df[[f"{cls}_pred" for cls in classes]].to_numpy()

            try:
                auc = roc_auc_score(
                    y_trues_group, y_probs_group, average="macro", multi_class="ovr"
                )
            except:
                auc = None

            overall_metrics = {
                "auc": auc,
                "accuracy": accuracy_score(y_trues_group, y_preds_group),
                "precision": precision_score(
                    y_trues_group, y_preds_group, average="macro", zero_division=0
                ),
                "recall": recall_score(
                    y_trues_group, y_preds_group, average="macro", zero_division=0
                ),
                "num_samples": len(group_df),
            }

            class_metrics = []
            tpr_list, fpr_list, fnr_list, tnr_list, ppr_list = [], [], [], [], []

            for i, class_name in enumerate(classes):
                y_trues_c = y_trues_group[:, i]
                y_preds_c = y_preds_group[:, i]

                tn, fp, fn, tp = confusion_matrix(
                    y_trues_c, y_preds_c, labels=[0, 1]
                ).ravel()

                tpr = tp / (tp + fn) if (tp + fn) > 0 else 0  # Sensitivity
                fpr = fp / (fp + tn) if (fp + tn) > 0 else 0  # Fall-out
                fnr = fn / (fn + tp) if (fn + tp) > 0 else 0  # Miss rate
                tnr = tn / (tn + fp) if (tn + fp) > 0 else 0  # Specificity

                ppr = (tp + fp) / len(y_trues_c)  # Positive Prediction Rate
                class_distribution = np.mean(y_preds_c)

                class_metrics.append(
                    {
                        "class_name": class_name,
                        "tpr": tpr,
                        "fpr": fpr,
                        "fnr": fnr,
                        "tnr": tnr,
                        "ppr": ppr,
                        "class_distribution": class_distribution,
                    }
                )

                tpr_list.append(tpr)
                fpr_list.append(fpr)
                fnr_list.append(fnr)
                tnr_list.append(tnr)
                ppr_list.append(ppr)

            overall_metrics.update(
                {
                    "mean_tpr": np.mean(tpr_list),
                    "mean_fpr": np.mean(fpr_list),
                    "mean_fnr": np.mean(fnr_list),
                    "mean_tnr": np.mean(tnr_list),
                    "mean_ppr": np.mean(ppr_list),
                }
            )

            group_metrics[group] = {
                "overall_metrics": overall_metrics,
                "class_metrics": class_metrics,
            }

        results[attr] = group_metrics

    return results


def get_fairness_metrics(group_metrics):
    summary = {}

    for attr, group_data in group_metrics.items():
        tprs = {}
        fprs = {}
        aucs = {}

        for group, metrics in group_data.items():
            tprs[group] = metrics["overall_metrics"].get("mean_tpr", 0)
            fprs[group] = metrics["overall_metrics"].get("mean_fpr", 0)
            aucs[group] = metrics["overall_metrics"].get("auc", 0)

        # Equal Opportunity: max TPR - min TPR
        max_tpr_group = max(tprs, key=tprs.get)
        min_tpr_group = min(tprs, key=tprs.get)
        equal_opportunity_diff = abs(tprs[max_tpr_group] - tprs[min_tpr_group])
        equal_opportunity_ratio = tprs[min_tpr_group] / tprs[max_tpr_group]

        # Equalized Odds: max(TPR + FPR) - min(TPR + FPR)
        odds = {g: tprs[g] + fprs[g] for g in tprs}
        max_odds_group = max(odds, key=odds.get)
        min_odds_group = min(odds, key=odds.get)
        equalized_odds_diff = abs(odds[max_odds_group] - odds[min_odds_group])
        equalized_odds_ratio = odds[min_odds_group] / odds[max_odds_group]

        # AUC GAP and Worst AUC
        max_auc_group = max(aucs, key=aucs.get)
        min_auc_group = min(aucs, key=aucs.get)
        auc_gap = aucs[max_auc_group] - aucs[min_auc_group]
        worst_auc = aucs[min_auc_group]

        summary[attr] = {
            "equal_opportunity_diff": {
                "value": equal_opportunity_diff,
                "groups": [min_tpr_group, max_tpr_group],
            },
            "equal_opportunity_ratio": {
                "value": equal_opportunity_ratio,
                "groups": [min_tpr_group, max_tpr_group],
            },
            "equalized_odds_diff": {
                "value": equalized_odds_diff,
                "groups": [min_odds_group, max_odds_group],
            },
            "equalized_odds_ratio": {
                "value": equalized_odds_ratio,
                "groups": [min_odds_group, max_odds_group],
            },
            "worst_auc": {
                "value": worst_auc,
                "group": min_auc_group,
            },
            "auc_gap": {
                "value": auc_gap,
                "groups": [min_auc_group, max_auc_group],
            },
        }

    return summary


def save_json_results(output_dir, filename, data):
    with open(os.path.join(output_dir, filename), "w") as f:
        json.dump(data, f)
