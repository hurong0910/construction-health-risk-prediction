\
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, accuracy_score, average_precision_score, precision_recall_curve, roc_curve, auc

def eval_from_preds(y_true, y_pred, target_names):
    rep = classification_report(y_true, y_pred, target_names=target_names, labels=list(range(len(target_names))), output_dict=True, zero_division=0)
    acc = accuracy_score(y_true, y_pred)
    macro_f1 = rep['macro avg']['f1-score']
    extreme_rec = rep[target_names[-1]]['recall']
    return dict(Accuracy=acc, Macro_F1=macro_f1, Extreme_Recall=extreme_rec), rep

def macro_auprc_from_proba(y_true, proba, classes):
    # y_true: (n,), proba: (n, C)
    Y = np.zeros((len(y_true), len(classes)), dtype=int)
    for i, c in enumerate(classes):
        Y[:, i] = (y_true == c).astype(int)
    ap_list = [average_precision_score(Y[:, j], proba[:, j]) for j in range(len(classes))]
    return float(np.mean(ap_list)), ap_list

def plot_pr_curves_by_class(y_true, proba, class_names, out_png):
    Y = np.zeros((len(y_true), len(class_names)), dtype=int)
    for j in range(len(class_names)):
        Y[:, j] = (y_true == j).astype(int)
    plt.figure(figsize=(6,4))
    for j, name in enumerate(class_names):
        p, r, _ = precision_recall_curve(Y[:, j], proba[:, j])
        ap = average_precision_score(Y[:, j], proba[:, j])
        plt.plot(r, p, label=f"{name} (AP={ap:.2f})")
    plt.xlabel("Recall"); plt.ylabel("Precision")
    plt.title("Precision–Recall Curves by Class")
    plt.legend(loc="lower left")
    plt.tight_layout(); plt.savefig(out_png, dpi=300); plt.close()
