from sklearn.metrics import roc_auc_score, precision_recall_curve, auc, roc_curve, confusion_matrix
from tqdm import tqdm
from models import get_model_instance
import numpy as np
import torch

def sigmoid(scores):
    return 1/(1 + np.exp(-scores))

def youden_score(y_true, y_preds):

    tn, fp, fn, tp = confusion_matrix(y_true, y_preds).ravel()

    # Then calculate specificity
    specificity = tn / (tn+fp)
    # Calculate sensitivity
    sensitivity = tp / (tp+fn)

    return sensitivity, specificity, sensitivity + specificity - 1

def calculate_youden_threshold(y_true, y_scores):
    """
    Calculate the threshold that maximizes Youden's J index.

    Args:
    - y_true (np.array): True binary labels.
    - y_scores (np.array): Target scores, probability estimates of the positive class.

    Returns:
    - tuple: Maximum Youden's J index and the corresponding optimal threshold.
    """
    # Calculate the false positive rate, true positive rate and thresholds
    fpr, tpr, thresholds = roc_curve(y_true, y_scores)

    # Calculate Youden's J index for each threshold
    youden_index = tpr - fpr

    # Find the optimal threshold
    optimal_idx = np.argmax(youden_index)
    optimal_threshold = thresholds[optimal_idx]

    return youden_index[optimal_idx], optimal_threshold

def calculate_metrics(y_true, y_scores, mode='auprc', patient_idxs=None):
    """
    Calculate evaluation metrics for binary classification.

    Parameters:
    - y_true (array-like): True labels of the samples.
    - y_scores (array-like): Predicted scores or probabilities of the positive class.
    - mode (str, optional): Evaluation mode. Can be 'auroc' for Area Under the Receiver Operating Characteristic Curve
      or 'auprc' for Area Under the Precision-Recall Curve. Default is 'auprc'.
    - patient_idxs (array-like, optional): Indexes indicating the patients to calculate metrics separately.
      If provided, metrics will be calculated for each patient individually. Default is None.

    Returns:
    - score (float): Evaluation score based on the selected mode.
    """
    if patient_idxs is not None:
        patient_aucs = []
        
        for unique_patient_idx in np.unique(patient_idxs):
            patient_true = np.concatenate(y_true)[patient_idxs == unique_patient_idx]
            patient_scores = np.concatenate(y_scores)[patient_idxs == unique_patient_idx]

            if mode == 'auroc':
                # Calculate AUC for current patient
                patient_aucs.append(roc_auc_score(patient_true, patient_scores))

            elif mode == 'auprc':
                # Calculate AUPRC for current patient
                precision, recall, _ = precision_recall_curve(patient_true, patient_scores)
                patient_aucs.append(auc(recall, precision))

        score = np.mean(patient_aucs)
    else:
        if mode == 'auroc':
            score = roc_auc_score(np.concatenate(y_true), np.concatenate(y_scores))
        elif mode == 'auprc':
            # Calculate AUPRC
            precision, recall, _ = precision_recall_curve(np.concatenate(y_true), np.concatenate(y_scores))
            score = auc(recall, precision)
    return score

def evaluate_model(model, loader, device, mode='test', youden_threshold=0.5):
    """
    Evaluate the performance of a model on a given dataset.

    Args:
        model (torch.nn.Module): The trained model to evaluate.
        loader (torch.utils.data.DataLoader): The data loader for the dataset.
        device (torch.device): The device to perform the evaluation on.
        mode (str, optional): The evaluation mode. Can be 'test' or 'validation'. Defaults to 'test'.
        youden_threshold (float, optional): The threshold value for calculating Youden's index. Defaults to 0.5.

    Returns:
        dict: A dictionary containing the aggregated evaluation metrics, including AUROC, AUPRC, baseline, Youden's index, specificity, sensitivity, and Youden's threshold.
    """

    if mode == 'validation':
        y_true, y_scores, patient_idxs = get_preds(model, loader, device, mode=mode)
    elif mode == 'test':
        y_true, y_scores, _, _, patient_idxs = get_preds(model, loader, device, mode=mode)
    else:
        raise ValueError(f"Invalid mode: {mode}")
    
    patient_aucs = []
    patient_auprcs = []
    youdens = []
    baselines = []
    specificities = []
    sensitivities = []

    # Calculate overall AUROC
    overall_auc = roc_auc_score(y_true, y_scores)

    # Calculate overall AUPRC
    precision, recall, _ = precision_recall_curve(y_true, y_scores)
    overall_auprc = auc(recall, precision)

    if mode == 'validation':
        _, youden_threshold = calculate_youden_threshold(y_true, y_scores)

    for unique_patient_idx in np.unique(patient_idxs):
        patient_true = y_true[patient_idxs == unique_patient_idx]
        patient_scores = y_scores[patient_idxs == unique_patient_idx]

        # Calculate AUC for current patient
        patient_aucs.append(roc_auc_score(patient_true, patient_scores))

        # Calculate AUPRC for current patient
        precision, recall, _ = precision_recall_curve(patient_true, patient_scores)
        patient_auprcs.append(auc(recall, precision))

        # Calculate baseline for current patient
        baselines.append(np.mean(patient_true))

        sensitivity, specificity, youden = youden_score(patient_true, patient_scores > youden_threshold)

        specificities.append(specificity)
        sensitivities.append(sensitivity)
        youdens.append(youden)

    # Return aggregated metrics
    return {
        'AUROC (averaged)': np.mean(patient_aucs),
        'AUPRC (averaged)': np.mean(patient_auprcs),
        'AUROC (all)': overall_auc,
        'AUPRC (all)': overall_auprc,
        'Baseline (averaged)': np.mean(baselines),
        'Baseline (all)': np.mean(y_true),
        'Youden (averaged)': np.mean(youdens),
        'Specificity (averaged)': np.mean(specificities),
        'Sensitivity (averaged)': np.mean(sensitivities),
        'Youden threshold': youden_threshold
    }


def get_preds(model, loader, device, mode='test'):
    """
    Obtain predictions from the model using the loader.

    Parameters:
    - model: The neural network model to evaluate.
    - loader: DataLoader for the dataset.
    - device: The device to run the model on.
    - mode: Mode of operation ('test' or other).

    Returns:
    - Tuple of numpy arrays: (y_true, y_scores, (coords), (lobes), patient_idxs).
      Coordinates and lobes are included only in 'test' mode.
    """
    if mode not in ['test', 'train', 'validation']:
        raise ValueError("Invalid mode. Expected 'train', 'validation', or 'test'.")

    model.eval()
    y_true, y_scores, coords, lobes, patient_idxs = [], [], [], [], []

    with torch.no_grad():
        for *inputs, labels, patient_idx in tqdm(loader):
            inputs = [input_.to(device) for input_ in inputs]
            labels = labels.to(device)

            outputs = model(inputs[:2] if mode == 'test' else inputs)[:, 0]

            y_true.append(labels.cpu().detach().numpy())
            y_scores.append(outputs.cpu().detach().numpy())
            if mode == 'test':
                coords.append(inputs[2].cpu().detach().numpy())
                lobes.append(inputs[3].cpu().detach().numpy())
            patient_idxs.append(patient_idx.cpu().detach().numpy())

    if mode == 'test':
        return (np.concatenate(y_true), sigmoid(np.concatenate(y_scores)), 
                np.concatenate(coords), np.concatenate(lobes), 
                np.concatenate(patient_idxs))
    else:
        return np.concatenate(y_true), sigmoid(np.concatenate(y_scores)), np.concatenate(patient_idxs)


def get_thresh_and_evaluate(model, device, val_loader, test_loader):

    # Evaluate the model
    val_metrics = evaluate_model(model, val_loader, device, mode='validation')

    # Get F1 threshold from validation set
    youden_threshold = val_metrics['Youden threshold']

    # Evaluate the model
    test_metrics = evaluate_model(model, test_loader, device, mode='test', youden_threshold=youden_threshold)

    return test_metrics
