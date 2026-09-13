import math

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf


def load_test_dataset(directory, image_size=(128, 128), batch_size=32):
    dataset = tf.keras.utils.image_dataset_from_directory(
        directory,
        image_size=image_size,
        batch_size=batch_size,
        shuffle=False,
    )
    return dataset, dataset.class_names


def collect_predictions(model, dataset):
    labels = []
    probabilities = []
    for images, batch_labels in dataset:
        outputs = model(images, training=False)
        probabilities.append(np.asarray(outputs).reshape(-1))
        labels.append(np.asarray(batch_labels).reshape(-1))
    return np.concatenate(labels).astype(int), np.concatenate(probabilities).astype(float)


def confusion_matrix(y_true, y_pred, num_classes=2):
    matrix = np.zeros((num_classes, num_classes), dtype=int)
    np.add.at(matrix, (np.asarray(y_true, dtype=int), np.asarray(y_pred, dtype=int)), 1)
    return matrix


def _proportion_interval(proportion, count):
    if count == 0:
        return 0.0
    return 1.96 * math.sqrt(proportion * (1 - proportion) / count)


def per_class_report(y_true, y_prob, class_names, threshold=0.5):
    y_true = np.asarray(y_true, dtype=int)
    y_pred = (np.asarray(y_prob) > threshold).astype(int)
    matrix = confusion_matrix(y_true, y_pred, len(class_names))
    classes = []
    for index, name in enumerate(class_names):
        support = int(matrix[index].sum())
        correct = int(matrix[index, index])
        predicted = int(matrix[:, index].sum())
        recall = correct / support if support else 0.0
        precision = correct / predicted if predicted else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        classes.append({
            "class": name,
            "support": support,
            "correct": correct,
            "recall": recall,
            "recall_interval": _proportion_interval(recall, support),
            "precision": precision,
            "f1": f1,
        })
    total = int(matrix.sum())
    report = {
        "threshold": threshold,
        "matrix": matrix,
        "classes": classes,
        "accuracy": int(np.trace(matrix)) / total if total else 0.0,
        "balanced_accuracy": float(np.mean([item["recall"] for item in classes])),
        "macro_f1": float(np.mean([item["f1"] for item in classes])),
        "class_counts": {item["class"]: item["support"] for item in classes},
    }
    if len(classes) == 2:
        first, second = classes
        report["gap"] = abs(first["recall"] - second["recall"])
        report["gap_noise"] = math.sqrt(
            _proportion_interval(first["recall"], first["support"]) ** 2
            + _proportion_interval(second["recall"], second["support"]) ** 2
        )
        report["gap_significant"] = report["gap"] > report["gap_noise"]
    return report


def check_release_targets(report, previous=None, min_class_recall=0.93, max_gap=0.03, max_class_drop=0.01, min_balance=0.9):
    results = []
    for item in report["classes"]:
        results.append((f"{item['class']} recall", item["recall"], min_class_recall, item["recall"] >= min_class_recall))
    if "gap" in report:
        results.append(("gap between classes", report["gap"], max_gap, report["gap"] <= max_gap))
    counts = list(report["class_counts"].values())
    balance = min(counts) / max(counts) if max(counts) else 0.0
    results.append(("test set balance (smallest / largest class)", balance, min_balance, balance >= min_balance))
    if previous:
        for item in report["classes"]:
            if item["class"] in previous:
                drop = previous[item["class"]] - item["recall"]
                results.append((f"{item['class']} recall drop vs previous", drop, max_class_drop, drop <= max_class_drop))
    return results


def plot_confusion_matrix(matrix, class_names):
    matrix = np.asarray(matrix)
    row_totals = matrix.sum(axis=1, keepdims=True)
    shares = np.divide(matrix, row_totals, out=np.zeros(matrix.shape, dtype=float), where=row_totals > 0)
    fig, ax = plt.subplots(figsize=(5, 4), layout="constrained")
    image = ax.imshow(shares, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(len(class_names)), labels=class_names)
    ax.set_yticks(range(len(class_names)), labels=class_names)
    ax.set_xlabel("predicted")
    ax.set_ylabel("true")
    ax.set_title("confusion matrix")
    for row in range(matrix.shape[0]):
        for column in range(matrix.shape[1]):
            color = "white" if shares[row, column] > 0.5 else "black"
            ax.text(column, row, f"{matrix[row, column]}\n{shares[row, column]:.1%}", ha="center", va="center", color=color)
    fig.colorbar(image, ax=ax, label="share of true class")
    return fig
