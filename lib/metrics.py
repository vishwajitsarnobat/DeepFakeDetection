# lib/metrics.py
import os
import numpy as np
from sklearn import metrics
import torch
# Import necessary sklearn metrics
from sklearn.metrics import accuracy_score, roc_auc_score, average_precision_score, recall_score, f1_score

# Assuming these utility functions exist elsewhere and handle PyTorch tensors
# from losses.losses import _avg_sigmoid, _sigmoid

# --- Placeholder for sigmoid if losses.losses is unavailable ---
def _sigmoid(x):
    """Applies sigmoid activation."""
    if isinstance(x, torch.Tensor):
        return torch.sigmoid(x)
    elif isinstance(x, np.ndarray):
        return 1 / (1 + np.exp(-x))
    else:
        raise TypeError("Input must be torch.Tensor or np.ndarray")

def _softmax(x, dim=-1):
     """Applies softmax activation."""
     if isinstance(x, torch.Tensor):
        return torch.softmax(x, dim=dim)
     elif isinstance(x, np.ndarray):
        e_x = np.exp(x - np.max(x, axis=dim, keepdims=True))
        return e_x / np.sum(e_x, axis=dim, keepdims=True)
     else:
        raise TypeError("Input must be torch.Tensor or np.ndarray")

def _avg_sigmoid(x):
     """Placeholder: Applies sigmoid then averages spatial dimensions if needed."""
     # This function's original purpose might be specific to heatmap averaging.
     # Assuming input is a tensor BxCxHxW or similar
     if isinstance(x, torch.Tensor):
          if x.ndim > 2: # Assume spatial dims exist
               return torch.mean(_sigmoid(x), dim=tuple(range(2, x.ndim)))
          else: # Apply sigmoid if no spatial dims (e.g., BxC)
               return _sigmoid(x)
     else: # Handle numpy array similarly
          if x.ndim > 2:
               return np.mean(_sigmoid(x), axis=tuple(range(2, x.ndim)))
          else:
               return _sigmoid(x)


# =====================================================================
# --- Binary Classification Metrics (Using NumPy Inputs) ---
# =====================================================================

def bin_calculate_acc(preds_np, labels_np=None, targets=None, threshold=0.5, **kwargs):
    """
    Calculate binary accuracy using NumPy.

    Args:
        preds_np (np.ndarray): Numpy array of prediction scores (prob class 1) or logits. Shape (N,).
        labels_np (np.ndarray): Numpy array of ground truth labels (0 or 1). Shape (N,).
        targets: Ignored in this implementation.
        threshold (float): Threshold to convert scores to binary predictions.

    Returns:
        float: Accuracy score.
    """
    if labels_np is None:
        raise ValueError("Labels (labels_np) cannot be None for accuracy calculation.")

    # Ensure inputs are flat numpy arrays
    preds_flat = np.asarray(preds_np).ravel()
    labels_flat = np.asarray(labels_np).ravel()

    if preds_flat.shape != labels_flat.shape:
        raise ValueError(f"Shape mismatch: predictions ({preds_flat.shape}) vs labels ({labels_flat.shape})")

    # Apply sigmoid if predictions look like logits (have values outside [0,1])
    # Simple check: if any value is < 0 or > 1, assume logits
    if np.any(preds_flat < 0) or np.any(preds_flat > 1):
         print("Warning: Input predictions seem to be logits, applying sigmoid.")
         preds_flat = _sigmoid(preds_flat) # Apply element-wise sigmoid

    # Apply threshold to get binary predictions
    binary_preds = (preds_flat >= threshold).astype(int)

    # Calculate accuracy using sklearn
    try:
        acc = accuracy_score(labels_flat, binary_preds)
    except Exception as e:
        print(f"Error calculating accuracy: {e}")
        acc = 0.0

    return acc

def bin_calculate_auc_ap_ar(preds_np, labels_np, metrics_base='binary', hm_preds=None, cls_lamda=0.1, threshold=0.5, **kwargs):
    """
    Calculate AUC, AP, AR, and F1 score using NumPy and sklearn.

    Args:
        preds_np (np.ndarray): Numpy array of prediction scores (prob class 1) or logits. Shape (N,).
        labels_np (np.ndarray): Numpy array of ground truth labels (0 or 1). Shape (N,).
        metrics_base (str): Specifies how predictions were derived ('binary', 'heatmap', 'combine').
        hm_preds (np.ndarray, optional): Heatmap predictions if metrics_base is 'combine'.
        cls_lamda (float, optional): Lambda for combining heatmap and class scores.
        threshold (float): Threshold for calculating AR and F1.

    Returns:
        tuple: (AUC, AP, AR, F1) scores.
    """
    if labels_np is None:
        raise ValueError("Labels (labels_np) cannot be None.")

    # Ensure inputs are flat numpy arrays
    cls_preds_flat = np.asarray(preds_np).ravel()
    labels_flat = np.asarray(labels_np).ravel()

    if cls_preds_flat.shape != labels_flat.shape:
        raise ValueError(f"Shape mismatch: preds ({cls_preds_flat.shape}) vs labels ({labels_flat.shape})")

    # --- Combine predictions if needed ---
    if metrics_base == 'combine':
        if hm_preds is None:
             raise ValueError('Heatmap predictions (hm_preds) required when metrics_base is "combine".')
        try:
             hm_preds_np = np.asarray(hm_preds)
             # Assuming hm_preds are BxCxHxW or similar, apply sigmoid and average/pool
             # This part needs to match how it was done during training/previous steps
             # Example: Simple spatial max pooling after sigmoid
             hm_scores = np.max(_sigmoid(hm_preds_np).reshape(hm_preds_np.shape[0], -1), axis=1)
             # Combine scores
             final_preds_flat = cls_lamda * cls_preds_flat + (1 - cls_lamda) * hm_scores.ravel()
             print("Info: Combining heatmap and classification scores for metrics.")
        except Exception as e:
             print(f"Error combining heatmap and class predictions: {e}. Using class predictions only.")
             final_preds_flat = cls_preds_flat # Fallback
    else:
         final_preds_flat = cls_preds_flat

    # --- Apply sigmoid if predictions look like logits ---
    if np.any(final_preds_flat < 0) or np.any(final_preds_flat > 1):
         print("Warning: Combined predictions seem to be logits, applying sigmoid.")
         final_preds_flat = _sigmoid(final_preds_flat) # Apply sigmoid

    # --- Calculate Metrics using sklearn ---
    try:
        # Ensure labels are integers 0 or 1
        labels_int = labels_flat.astype(int)

        # Check for edge cases (e.g., only one class present)
        unique_labels = np.unique(labels_int)
        if len(unique_labels) < 2:
             print(f"Warning: Only one class ({unique_labels}) present in labels. AUC/AP undefined.")
             auc_, ap_ = 0.0, 0.0 # Or np.nan
        else:
             auc_ = roc_auc_score(labels_int, final_preds_flat)
             ap_ = average_precision_score(labels_int, final_preds_flat)

        # Calculate AR and F1 based on threshold
        binary_preds = (final_preds_flat >= threshold).astype(int)

        # Recall for the positive class (label 1)
        ar_ = recall_score(labels_int, binary_preds, pos_label=1, zero_division=0)

        # F1 score for the positive class (label 1)
        f1_ = f1_score(labels_int, binary_preds, pos_label=1, zero_division=0)

    except ValueError as ve: # Catch specific sklearn errors like "Only one class present"
         print(f"Error calculating sklearn metrics: {ve}. Check input shapes and label values.")
         auc_, ap_, ar_, f1_ = 0.0, 0.0, 0.0, 0.0 # Defaults on error
    except Exception as e:
         print(f"Unexpected error calculating metrics: {e}")
         auc_, ap_, ar_, f1_ = 0.0, 0.0, 0.0, 0.0 # Defaults on error


    return auc_, ap_, ar_, f1_


# =====================================================================
# --- Heatmap-based and Combined Accuracy (Require PyTorch Tensors) ---
# =====================================================================
# These functions likely expect PyTorch tensors as input because they
# use functions like _avg_sigmoid, _sigmoid, torch.reshape, torch.topk.
# They should ideally be called *before* converting predictions to NumPy.
# If called from test.py after numpy conversion, they will fail.

def hm_calculate_acc(hm_preds_tensor, labels_tensor=None, targets=None, threshold=0.5, **kwargs):
    """
    Calculate accuracy based on averaged heatmap predictions.
    EXPECTS PYTORCH TENSORS.
    """
    if not isinstance(hm_preds_tensor, torch.Tensor) or not isinstance(labels_tensor, torch.Tensor):
         print("Warning: hm_calculate_acc expects PyTorch tensors.")
         # Attempt conversion back, but this is inefficient
         hm_preds_tensor = torch.as_tensor(hm_preds_tensor)
         labels_tensor = torch.as_tensor(labels_tensor)
         # return 0.0 # Or handle differently

    # Assume _avg_sigmoid handles tensor input and spatial averaging -> (B, 1) or (B,) scores
    cls_scores_tensor = _avg_sigmoid(hm_preds_tensor)
    # Pass scores and labels (as tensors) to bin_calculate_acc_torch if available,
    # or convert to numpy *here* if bin_calculate_acc only takes numpy
    cls_scores_np = cls_scores_tensor.detach().cpu().numpy()
    labels_np = labels_tensor.detach().cpu().numpy()
    return bin_calculate_acc(cls_scores_np, labels_np=labels_np, threshold=threshold)


def hm_bin_calculate_acc(hm_preds_tensor, cls_preds_tensor, labels_tensor=None, targets=None, cls_lamda=0.05, threshold=0.5, **kwargs):
    """
    Calculate accuracy combining heatmap and classification predictions.
    EXPECTS PYTORCH TENSORS.
    """
    if not all(isinstance(t, torch.Tensor) for t in [hm_preds_tensor, cls_preds_tensor, labels_tensor]):
        print("Warning: hm_bin_calculate_acc expects PyTorch tensors.")
        # Attempt conversion back
        hm_preds_tensor = torch.as_tensor(hm_preds_tensor)
        cls_preds_tensor = torch.as_tensor(cls_preds_tensor)
        labels_tensor = torch.as_tensor(labels_tensor)
        # return 0.0

    try:
        # Apply sigmoid to classification predictions if they are logits
        if cls_preds_tensor.shape[-1] == 1 or torch.any(cls_preds_tensor < 0) or torch.any(cls_preds_tensor > 1):
             cls_preds_sig = _sigmoid(cls_preds_tensor)
        else: # Assume softmax was applied or it's multi-class prob [:, 1]
             cls_preds_sig = cls_preds_tensor[:, 1:2] if cls_preds_tensor.shape[-1] > 1 else cls_preds_tensor


        # Process heatmap predictions (e.g., top-k averaging)
        hm_preds_sig = _sigmoid(hm_preds_tensor) # BxCxHxW
        hm_preds_flat = torch.reshape(hm_preds_sig, (hm_preds_sig.shape[0], -1)) # Flatten spatial/channel dims BxN
        # Ensure K is not larger than the number of elements
        k = min(10, hm_preds_flat.shape[1])
        top_k_hm = torch.topk(hm_preds_flat, k, dim=-1).values # Get top k values BxK
        mean_hm_preds = torch.mean(top_k_hm, dim=-1, keepdim=True) # Average top k -> Bx1

        # Combine predictions
        combined_preds_tensor = cls_lamda * cls_preds_sig + (1 - cls_lamda) * mean_hm_preds

        # Calculate accuracy using the numpy-based function
        combined_preds_np = combined_preds_tensor.detach().cpu().numpy()
        labels_np = labels_tensor.detach().cpu().numpy()
        acc = bin_calculate_acc(combined_preds_np, labels_np=labels_np, threshold=threshold)

    except Exception as e:
        print(f"Error in hm_bin_calculate_acc: {e}")
        acc = 0.0

    return acc


# =====================================================================
# --- Function Selector ---
# =====================================================================

def get_acc_mesure_func(metrics_base='binary'):
    """
    Selects the appropriate accuracy calculation function based on metrics_base.
    Note: Ensure inputs match (NumPy vs Tensor) for the selected function.
    """
    if metrics_base == 'binary':
        # Returns function expecting NumPy arrays
        return bin_calculate_acc
    elif metrics_base == 'heatmap':
        # Returns function expecting PyTorch Tensors ideally
        # Might need wrapper if called with numpy in test.py
        print("Warning: 'heatmap' accuracy expects PyTorch tensors. Ensure correct input type.")
        return hm_calculate_acc
    elif metrics_base == 'combine':
         # Returns function expecting PyTorch Tensors ideally
         print("Warning: 'combine' accuracy expects PyTorch tensors. Ensure correct input type.")
         return hm_bin_calculate_acc
    else:
        print(f"Warning: Unknown metrics_base '{metrics_base}'. Defaulting to binary accuracy (expects NumPy).")
        return bin_calculate_acc