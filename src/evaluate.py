import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import (
    roc_auc_score, roc_curve,
    average_precision_score, precision_recall_curve,
    confusion_matrix, ConfusionMatrixDisplay,
    precision_score, recall_score, f1_score
)
from .config import FIGURES_DIR

def predict_proba(model, X):
    return model.predict_proba(X)[:, 1]

def save_roc_train_test(y_train, p_train, y_test, p_test, filename="roc_train_test.png"):
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    fpr_tr, tpr_tr, _ = roc_curve(y_train, p_train)
    fpr_te, tpr_te, _ = roc_curve(y_test, p_test)

    auc_tr = roc_auc_score(y_train, p_train)
    auc_te = roc_auc_score(y_test, p_test)

    plt.figure(figsize=(7, 5))
    plt.plot(fpr_tr, tpr_tr, linewidth=2, label=f"Train ROC (AUC={auc_tr:.3f})")
    plt.plot(fpr_te, tpr_te, linewidth=2, label=f"Test ROC (AUC={auc_te:.3f})")
    plt.plot([0, 1], [0, 1], linestyle="--", linewidth=1, label="Baseline")
    plt.title("ROC Curve — Train vs Test")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.legend()
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / filename, dpi=160)
    plt.close()

    return auc_tr, auc_te

def save_pr_train_test(y_train, p_train, y_test, p_test, filename="pr_train_test.png"):
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    prec_tr, rec_tr, _ = precision_recall_curve(y_train, p_train)
    prec_te, rec_te, _ = precision_recall_curve(y_test, p_test)

    ap_tr = average_precision_score(y_train, p_train)
    ap_te = average_precision_score(y_test, p_test)

    baseline = float(np.mean(y_test))

    plt.figure(figsize=(7, 5))
    plt.plot(rec_tr, prec_tr, linewidth=2, label=f"Train PR (AP={ap_tr:.3f})")
    plt.plot(rec_te, prec_te, linewidth=2, label=f"Test PR (AP={ap_te:.3f})")
    plt.hlines(baseline, 0, 1, linestyles="--", linewidth=1.5, label=f"Baseline = {baseline:.3f}")
    plt.title("Precision–Recall Curve — Injury Risk Model")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.legend()
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / filename, dpi=160)
    plt.close()

    return ap_tr, ap_te, baseline

def metrics_at_threshold(y_true, proba, thr):
    y_pred = (proba >= thr).astype(int)
    return {
        "threshold": float(thr),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "cm": confusion_matrix(y_true, y_pred),
    }

def save_confusion_matrices(y_true, proba, thresholds=(0.5,), filename="confusion_matrices.png"):
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    thresholds = list(thresholds)
    n = len(thresholds)
    fig_w = max(6, 4 * n)

    fig, axes = plt.subplots(1, n, figsize=(fig_w, 4))
    if n == 1:
        axes = [axes]

    summaries = []
    for ax, thr in zip(axes, thresholds):
        m = metrics_at_threshold(y_true, proba, thr)
        summaries.append({k: v for k, v in m.items() if k != "cm"})

        disp = ConfusionMatrixDisplay(confusion_matrix=m["cm"], display_labels=["No Injury", "Injury"])
        disp.plot(ax=ax, values_format="d", colorbar=False)
        ax.set_title(f"Threshold = {thr:.3f}")
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")

    plt.tight_layout()
    plt.savefig(FIGURES_DIR / filename, dpi=160)
    plt.close()
    return summaries