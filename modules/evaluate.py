"""Đánh giá mô hình: metric, bảng so sánh, ma trận nhầm lẫn, biểu đồ."""
from typing import Dict, Optional, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
)


def compute_metrics(y_true, y_pred) -> dict:
    """Accuracy, Macro Precision/Recall/F1 và Weighted F1 (đơn vị %)."""
    p, r, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="macro", zero_division=0)
    _, _, f1_w, _ = precision_recall_fscore_support(
        y_true, y_pred, average="weighted", zero_division=0
    )
    return {
        "Accuracy (%)": accuracy_score(y_true, y_pred) * 100,
        "Macro Precision (%)": p * 100,
        "Macro Recall (%)": r * 100,
        "Macro F1 (%)": f1 * 100,
        "Weighted F1 (%)": f1_w * 100,
    }


def print_report(y_true, y_pred, title: str = "", target_names: Optional[Sequence[str]] = None):
    """In classification_report của một mô hình."""
    if title:
        print("=" * 10, title, "=" * 10)
    print(classification_report(y_true, y_pred, target_names=target_names, zero_division=0))


def compare_models(y_true, results: Dict[str, dict]) -> pd.DataFrame:
    """Bảng so sánh các mô hình.

    ``results``: {"Tên mô hình": {"features": "TF-IDF", "y_pred": mảng dự đoán}}
    """
    rows = []
    for name, res in results.items():
        row = {"Model": name, "Features": res["features"]}
        row.update(compute_metrics(y_true, res["y_pred"]))
        rows.append(row)
    return pd.DataFrame(rows).round(2)


def plot_accuracy_comparison(df_compare: pd.DataFrame, save_path=None) -> None:
    """Biểu đồ cột so sánh Accuracy, tô màu theo phương pháp trích xuất đặc trưng."""
    plt.figure(figsize=(11, 5))
    sns.set_theme(style="whitegrid")
    ax = sns.barplot(
        data=df_compare,
        x="Model",
        y="Accuracy (%)",
        hue="Features",
        palette="Set2",
        dodge=False,
    )
    for p in ax.patches:
        h = p.get_height()
        if h > 0:
            ax.annotate(
                f"{h:.2f}%",
                (p.get_x() + p.get_width() / 2.0, h),
                ha="center",
                va="bottom",
                fontsize=9,
                fontweight="bold",
            )
    low = max(0, df_compare["Accuracy (%)"].min() - 10)
    ax.set_ylim(low, 105)
    ax.set_title("So sánh Accuracy giữa các mô hình", fontsize=13, fontweight="bold")
    ax.set_xlabel("")
    plt.xticks(rotation=20, ha="right")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()


def plot_confusion_matrices(
    y_true, preds: Dict[str, np.ndarray], labels: Sequence[str], ncols: int = 3, save_path=None
) -> None:
    """Lưới ma trận nhầm lẫn (chuẩn hóa theo hàng) cho nhiều mô hình."""
    n = len(preds)
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(5.2 * ncols, 4.4 * nrows), squeeze=False)
    for ax in axes.ravel():
        ax.axis("off")
    for ax, (name, y_pred) in zip(axes.ravel(), preds.items()):
        ax.axis("on")
        cm = confusion_matrix(y_true, y_pred, normalize="true")
        sns.heatmap(
            cm,
            annot=True,
            fmt=".2f",
            cmap="Blues",
            xticklabels=labels,
            yticklabels=labels,
            cbar=False,
            ax=ax,
        )
        ax.set_title(name, fontsize=11)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")
        ax.tick_params(axis="x", rotation=45)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()


def plot_training_curves(histories: Dict[str, dict], save_path=None) -> None:
    """Đường loss / accuracy theo epoch của các MLP."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    for name, h in histories.items():
        ax1.plot(h["loss"], label=name)
        ax2.plot(h["accuracy"], label=name)
    ax1.set_title("Training loss")
    ax2.set_title("Training accuracy")
    for ax in (ax1, ax2):
        ax.set_xlabel("Epoch")
        ax.legend()
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()
