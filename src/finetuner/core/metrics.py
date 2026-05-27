"""Module for calculating binary classification validation metrics."""

from typing import Any, Dict, List


class MetricsCalculator:
    """Computes standard binary classification performance metrics.

    High level role: Performance evaluator and metric calculator.
    Description: This class calculates standard classification metrics including
    Accuracy, Precision, Recall, and F1 Score from true validation labels and
    predicted unsafe probabilities, compiling them into a unified stats dictionary.

    How it works:
    1. Iterates over records and predicted probabilities.
    2. Compares if the true label is unsafe (equals 1) against the threshold-based prediction.
    3. Aggregates counts for True Positives, False Positives, True Negatives, and False Negatives.
    4. Computes ratio metrics safely, protecting against division by zero.
    """

    def compute_metrics(
        self,
        records: List[Dict[str, Any]],
        unsafe_probabilities: List[float],
        threshold: float = 0.5,
    ) -> Dict[str, Any]:
        """Calculates global metrics (Accuracy, F1, Precision, Recall) from unsafe probabilities.

        Args:
            records (List[Dict[str, Any]]): validation records containing true 'label' keys.
            unsafe_probabilities (List[float]): Predicted unsafe probabilities.
            threshold (float): Decision threshold to flag unsafe. Default is 0.5.

        Returns:
            Dict[str, Any]: Compiled metrics dictionary with accuracy, precision, recall, f1_score,
                tp, fp, tn, fn.
        """
        total = len(records)
        tp, fp, tn, fn = self._count_rates(records, unsafe_probabilities, threshold)
        acc = (tp + tn) / total if total > 0 else 0.0
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * (prec * rec) / (prec + rec) if (prec + rec) > 0 else 0.0

        return {
            "accuracy": acc,
            "precision": prec,
            "recall": rec,
            "f1_score": f1,
            "total_evaluated": total,
            "tp": tp,
            "fp": fp,
            "tn": tn,
            "fn": fn,
        }

    def _count_rates(
        self,
        records: List[Dict[str, Any]],
        unsafe_probabilities: List[float],
        threshold: float,
    ) -> tuple[int, int, int, int]:
        """Counts true/false positives and negatives from records.

        High level role: Confusion matrix rates aggregator.

        Args:
            records (List[Dict[str, Any]]): Input records.
            unsafe_probabilities (List[float]): Predicted unsafe probabilities.
            threshold (float): Safety threshold.

        Returns:
            tuple[int, int, int, int]: tuple of (tp, fp, tn, fn) counts.
        """
        tp, fp, tn, fn = 0, 0, 0, 0
        for r, pred_prob in zip(records, unsafe_probabilities, strict=False):
            true_unsafe = r.get("label", 0) == 1
            pred_unsafe = pred_prob >= threshold

            tp += int(true_unsafe and pred_unsafe)
            fp += int(not true_unsafe and pred_unsafe)
            tn += int(not true_unsafe and not pred_unsafe)
            fn += int(true_unsafe and not pred_unsafe)
        return tp, fp, tn, fn
