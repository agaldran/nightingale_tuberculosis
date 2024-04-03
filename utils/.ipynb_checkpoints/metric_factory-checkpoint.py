from sklearn import metrics
import math
from sklearn.metrics import roc_auc_score, roc_curve, balanced_accuracy_score, precision_recall_curve, auc, f1_score
from sklearn.metrics import average_precision_score

def get_imb_metrics(y_true, y_prob, thresh=0.5):
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    sens, spec = tpr, 1 - fpr
    sens_at_95_spec = sens[spec > 0.95][-1]
    precision, recall, thresholds = precision_recall_curve(y_true, y_prob)
    au_prc = auc(recall, precision)
    ap = average_precision_score(y_true, y_prob)
    # p_auc = roc_auc_score(y_true, y_prob, max_fpr=0.10)  # partial AUC (90-100% specificity)
    # bacc = balanced_accuracy_score(y_true, y_prob > thresh)
    # f1 = f1_score(y_true, y_prob)
    # return au_prc, sens_at_95_spec, p_auc, bacc, f1

    return au_prc, ap, sens_at_95_spec


