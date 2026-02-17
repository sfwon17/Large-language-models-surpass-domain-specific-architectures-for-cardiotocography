import torch 
import numpy as np

def downsample_4hz_to_1hz(data_array):
    n_samples, n_timesteps = data_array.shape
    
    # Trim to make divisible by 4
    trimmed_timesteps = (n_timesteps // 4) * 4
    trimmed_data = data_array[:, :trimmed_timesteps]
    
    # Reshape to group every 4 consecutive values and take mean
    reshaped = trimmed_data.reshape(n_samples, -1, 4)
    downsampled = np.mean(reshaped, axis=2)
    
    return downsampled

def min_max_standardize(data, min_val=-1, max_val=240.0):
    standardized_data = (data - min_val) / (max_val - min_val)
    return standardized_data

# simplified metrics calculations 
def calculate_metrics(y_true, y_pred, y_probs): 
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    y_probs = np.array(y_probs)

    accuracy = accuracy_score(y_true, y_pred)
    auc = roc_auc_score(y_true, y_probs)

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    sensitivity = tp / (tp + fn)
    specificity = tn / (tn + fp)

    return accuracy, auc, sensitivity, specificity
