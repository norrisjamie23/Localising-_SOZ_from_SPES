import numpy as np
import glob
import os
import torch
from torch.utils.data import DataLoader, TensorDataset

from sklearn.utils.class_weight import compute_class_weight
from sklearn.model_selection import KFold

def load_and_pad(file_path, max_dim2):
    """Load the numpy array from the given path and pad its second dimension."""
    arr = np.load(file_path)
    pad_width = [(0, 0) if i != 1 else (0, max_dim2 - arr.shape[1]) for i in range(3)]
    return np.pad(arr, pad_width, mode='constant')

def paths_by_replace(paths, pattern, replacement):
    """Replace the given pattern in each path with the given replacement."""
    return [path.replace(pattern, replacement) for path in paths]

def get_split_paths(paths, split_indices, fold):
    """Get the paths for the given fold."""

    return {
        'train': np.array(paths)[split_indices[fold]['train']],
        'val': np.array(paths)[split_indices[fold]['val']],
        'test': np.array(paths)[split_indices[fold]['test']]
    }

def X_from_paths(X_file_paths, mean=None, std=None, chans=None):
    """Load X from the given file paths and normalize separately."""
    # Load arrays to determine the maximum size along the second dimension
    arrays = [np.load(path) for path in X_file_paths]

    max_dim2 = max(arr.shape[1] for arr in arrays) if chans is None else chans

    # Load and pad each array, then stack them
    padded_arrays = [load_and_pad(path, max_dim2) for path in X_file_paths]
    X = np.vstack(padded_arrays)

    # Separate the first column (distance) and the remaining columns (time series)
    distances = X[:, :, -1]
    time_series = X[:, :, :-1]

    if mean is None and std is None:
        # Calculate mean and std for distances
        mean_dist = distances[distances > 13].mean()
        std_dist = distances[distances > 13].std()

        # Calculate mean and std for time series
        arrays_for_standardisation = [array[:, :, :-1] for array in arrays]
        mean_ts = np.mean([array[array.std(axis=2) > 0].mean() for array in arrays_for_standardisation])
        std_ts = np.mean([array.std(axis=2)[array.std(axis=2) > 0].mean() for array in arrays_for_standardisation])

    else:
        mean_dist, mean_ts = mean
        std_dist, std_ts = std

    # Normalize distances
    distances[distances > 13] = (distances[distances > 13] - mean_dist) / std_dist

    # Normalize time series
    mask = (np.std(time_series, axis=2, keepdims=False) != 0)
    time_series[mask] = (time_series[mask] - mean_ts) / std_ts

    # Combine distances and time series back into X
    X[:, :, 0] = distances
    X[:, :, 1:] = time_series

    X = X[:, np.newaxis]

    return X, (mean_dist, mean_ts), (std_dist, std_ts)

def get_splits(paths, n_splits=5, seed=0):

    # Initialize KFold
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)

    # Container for storing indices of all splits
    split_indices = []

    splits = list(kf.split(paths))

    for i in range(n_splits):
        # Further split the training set into train and validation sets
        test_idx = splits[i][1]
        val_idx = splits[(i + 1) % n_splits][1]

        train_idx = np.array(list(set(splits[i][0]) - set(val_idx)))

        # Store the indices for train, validation, and test sets
        split_indices.append({'train': train_idx, 'val': val_idx, 'test': test_idx})

    return split_indices

def create_dataset(mean_pattern, std_pattern, fold, seed, batch_size=8):

    # Specify your directory path
    stim_pattern = os.path.join('../' + mean_pattern, 'X_stim*.npy')

    # Get file paths
    X_stim_mean_paths = glob.glob(stim_pattern)
    X_stim_sd_paths = paths_by_replace(X_stim_mean_paths, mean_pattern, std_pattern)
    X_recording_mean_paths = paths_by_replace(X_stim_mean_paths, 'X_stim', 'X_recording')
    X_recording_sd_paths = paths_by_replace(X_recording_mean_paths, mean_pattern, std_pattern)
    y_paths = paths_by_replace(X_stim_mean_paths, 'mean/X_stim', 'main/y')
    coords_paths = paths_by_replace(y_paths, 'y_', 'coords_')
    lobes_paths = paths_by_replace(y_paths, 'y_', 'lobes_')

    # Get split indices
    split_indices = get_splits(X_stim_mean_paths, n_splits=5, seed=seed)

    # Get paths for the given fold
    X_stim_mean_paths = get_split_paths(X_stim_mean_paths, split_indices, fold)
    X_stim_sd_paths = get_split_paths(X_stim_sd_paths, split_indices, fold)
    X_recording_mean_paths = get_split_paths(X_recording_mean_paths, split_indices, fold)
    X_recording_sd_paths = get_split_paths(X_recording_sd_paths, split_indices, fold)
    y_paths = get_split_paths(y_paths, split_indices, fold)
    coords_paths = get_split_paths(coords_paths, split_indices, fold)
    lobes_paths = get_split_paths(lobes_paths, split_indices, fold)

    # Get X and standardise using training set mean and standard deviation
    X_stim_mean_train, mean, std = X_from_paths(X_stim_mean_paths['train'])
    X_stim_mean_val, _, _ = X_from_paths(X_stim_mean_paths['val'], mean, std)
    X_stim_mean_test, _, _ = X_from_paths(X_stim_mean_paths['test'], mean, std)

    # Get X and standardise using training set mean and standard deviation
    X_stim_sd_train, mean, std = X_from_paths(X_stim_sd_paths['train'])
    X_stim_sd_val, _, _ = X_from_paths(X_stim_sd_paths['val'], mean, std)
    X_stim_sd_test, _, _ = X_from_paths(X_stim_sd_paths['test'], mean, std)

    # Get X and standardise using training set mean and standard deviation
    X_stim_train = np.concatenate([X_stim_mean_train, X_stim_sd_train], axis=1)
    X_stim_val = np.concatenate([X_stim_mean_val, X_stim_sd_val], axis=1)
    X_stim_test = np.concatenate([X_stim_mean_test, X_stim_sd_test], axis=1)

    # Get X and standardise using training set mean and standard deviation
    X_recording_mean_train, mean, std = X_from_paths(X_recording_mean_paths['train'])
    X_recording_mean_val, _, _ = X_from_paths(X_recording_mean_paths['val'], mean, std)
    X_recording_mean_test, _, _ = X_from_paths(X_recording_mean_paths['test'], mean, std)

    # Get X and standardise using training set mean and standard deviation
    X_recording_sd_train, mean, std = X_from_paths(X_recording_sd_paths['train'])
    X_recording_sd_val, _, _ = X_from_paths(X_recording_sd_paths['val'], mean, std)
    X_recording_sd_test, _, _ = X_from_paths(X_recording_sd_paths['test'], mean, std)

    # Get X and standardise using training set mean and standard deviation
    X_recording_train = np.concatenate([X_recording_mean_train, X_recording_sd_train], axis=1)
    X_recording_val = np.concatenate([X_recording_mean_val, X_recording_sd_val], axis=1)
    X_recording_test = np.concatenate([X_recording_mean_test, X_recording_sd_test], axis=1)

    # Load y from 3 filepaths and stack them
    y_train = np.concatenate([np.load(path) for path in y_paths['train']])
    y_val = np.concatenate([np.load(path) for path in y_paths['val']])
    y_test = np.concatenate([np.load(path) for path in y_paths['test']])

    # Load coords from filepaths and stack them
    coords_test = np.concatenate([np.load(path) for path in coords_paths['test']])
    lobes_test = np.concatenate([np.load(path) for path in lobes_paths['test']])

    # Patient index for each element of the test set
    val_indices = np.concatenate([[i] * np.load(X_stim_path).shape[0] for i, X_stim_path in enumerate(X_stim_mean_paths['val'])])
    test_indices = np.concatenate([[i] * np.load(X_stim_path).shape[0] for i, X_stim_path in enumerate(X_stim_mean_paths['test'])])

    test_indices_mapped = np.array([int(y_path[-6:-4]) for y_path in y_paths['test']])
    # map test indices to patient indices
    test_indices = test_indices_mapped[test_indices]

    # Create datasets
    train_dataset = TensorDataset(torch.from_numpy(X_stim_train).float(), torch.from_numpy(X_recording_train).float(), torch.from_numpy(y_train).float())
    val_dataset = TensorDataset(torch.from_numpy(X_stim_val).float(), torch.from_numpy(X_recording_val).float(), torch.from_numpy(y_val).float(), torch.from_numpy(val_indices).int())
    test_dataset = TensorDataset(torch.from_numpy(X_stim_test).float(), torch.from_numpy(X_recording_test).float(), torch.from_numpy(coords_test).float(), torch.from_numpy(lobes_test).int(), torch.from_numpy(y_test).float(), torch.from_numpy(test_indices).int())

    # Create the data loaders with the given batch size
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # Compute class weights
    if len(y_train.shape) > 1:
        weight = compute_class_weight('balanced', classes=np.unique(y_train.argmax(axis=1)), y=y_train.argmax(axis=1))
        weight = torch.tensor(weight, dtype=torch.float)

    # Assuming binary classification, get pos_weight
    else:
        weight = (y_train==0.).sum()/y_train.sum()

    # Return the data loaders and class weights
    return train_loader, val_loader, test_loader, weight
