import torch
from torch import Tensor
from sklearn.metrics import roc_auc_score
from medpy.metric import dc, hd95, asd


def get_confusion_matrix(Pred: Tensor, GT: Tensor, threshold: float = 0.5) -> tuple:
    """
    Calculates the confusion matrix for binary classification.

    Args:
        Pred (Tensor): The predicted segmentation results, before thresholding.
        GT (Tensor): The ground truth segmentation.
        threshold (float, optional): The threshold for binarization. Defaults to 0.5.

    Returns:
        tuple: True Positives, False Positives, False Negatives and True Negatives.
    """

    Pred = (torch.sigmoid(Pred.detach()) > threshold).float()
    GT = (GT == torch.max(GT.detach())).float()
    TP = (Pred * GT).sum().item()
    FP = (Pred * (1 - GT)).sum().item()
    FN = ((1 - Pred) * GT).sum().item()
    TN = ((1 - Pred) * (1 - GT)).sum().item()

    return TP, FP, FN, TN


def get_metrics(Pred: Tensor, GT: Tensor, threshold: float = 0.5) -> tuple:
    """
    Calculates the metrics for binary class using the confusion matrix.

    Args:
        Pred (Tensor): The predicted segmentation results, before thresholding.
        GT (Tensor): The ground truth segmentation.
        threshold (float, optional): The threshold for binarization. Defaults to 0.5.

    Returns:
        tuple: F1 score, Dice coefficient, Precision, Recall, Specificity, IoU, Accuracy.
    """
    smooth = 1e-6
    tp, fp, fn, tn = get_confusion_matrix(Pred, GT, threshold)

    precision = tp / (tp + fp + smooth)
    recall = tp / (tp + fn + smooth)
    specificity = tn / (tn + fp + smooth)
    dice = (2 * tp) / ((2 * tp) + fp + fn + smooth)
    f1 = (2 * precision * recall) / (precision + recall + smooth)
    iou = tp / (tp + fn + fp + smooth)
    acc = (tp + tn) / (tp + tn + fp + fn)

    return f1, dice, precision, recall, specificity, iou, acc


def get_metrics_auc(Pred: Tensor, GT: Tensor) -> float:
    """
    Calculates the AUC-ROC score for binary classification.

    Args:
        Pred (Tensor): The predicted segmentation results, before thresholding.
        GT (Tensor): The ground truth segmentation.

    Returns:
        float: The AUC-ROC score.
    """
    # Move tensors to CPU and convert to numpy arrays
    Pred = torch.sigmoid(Pred.detach()).cpu().numpy().flatten()
    GT = (GT == torch.max(GT.detach())).cpu().numpy().flatten()

    # Calculate AUC-ROC score
    auc = roc_auc_score(GT, Pred)

    return auc


def get_metrics_medpy(
    Pred: Tensor, GT: Tensor, threshold: float = 0.5
) -> (float, float, float):
    """
    Calculates the Dice Similarity Coefficient (DSC), 95th percentile Hausdorff Distance (HD95),
    and Average Surface Distance (ASD) using the MedPy library.

    Args:
        Pred (Tensor): The predicted segmentation results, before thresholding.
        GT (Tensor): The ground truth segmentation.
        threshold (float, optional): The threshold for binarization. Defaults to 0.5.

    Returns:
        tuple(float, float, float): The DSC, HD95, and ASD metrics.
    """
    Pred = (torch.sigmoid(Pred.detach()) > threshold).cpu().numpy()
    GT = (GT == torch.max(GT.detach())).cpu().numpy()

    # Calculate metrics using MedPy
    dice_ = dc(Pred, GT)
    asd_ = asd(Pred, GT)
    hd95_ = hd95(Pred, GT)

    return dice_, hd95_, asd_
