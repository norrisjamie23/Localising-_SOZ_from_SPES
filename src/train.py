from models import *
from dataset import create_dataset
import numpy as np
import torch.optim as optim
from tqdm import tqdm
from evaluate import calculate_metrics, evaluate_model, get_thresh_and_evaluate

def train_model(net, train_loader, val_loader, optimizer, criterion, save_path, device='mps', num_epochs=1000, patience=10):
    """
    Trains a model using the given data loaders, optimizer, and criterion.

    Args:
        net (torch.nn.Module): The model to be trained.
        train_loader (torch.utils.data.DataLoader): The data loader for the training set.
        val_loader (torch.utils.data.DataLoader): The data loader for the validation set.
        optimizer (torch.optim.Optimizer): The optimizer used for training.
        criterion (torch.nn.Module): The loss function used for training.
        save_path (str): The path to save the best model.
        device (str, optional): The device to use for training. Defaults to 'mps'.
        num_epochs (int, optional): The maximum number of epochs to train. Defaults to 1000.
        patience (int, optional): The number of epochs to wait for improvement in validation AUROC before early stopping. Defaults to 10.

    Returns:
        torch.nn.Module: The trained model.
    """
    save_best_model = SaveBestModel(save_path=save_path)

    for epoch in range(1, num_epochs + 1):
        net.train()  # Set model to training mode
        running_loss, y_true, y_scores = 0.0, [], []

        for *inputs, labels in tqdm(train_loader):
            inputs = [input.to(device) for input in inputs]
            
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = net(inputs)[:, 0]
            loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            y_true.append(labels.cpu().detach().numpy())
            y_scores.append(outputs.cpu().detach().numpy())

        epoch_train_auroc = calculate_metrics(y_true, y_scores, mode='auroc')

        print(f"Epoch {epoch}/{num_epochs}, Train Loss: {running_loss / len(train_loader):.4f}, Train AUROC: {epoch_train_auroc:.4f}")

        metrics = evaluate_model(net, val_loader, device=device, mode='validation')
        epoch_val_auroc = metrics['AUROC (averaged)']

        print(f"Epoch {epoch}/{num_epochs}, Valid AUROC: {epoch_val_auroc:.4f}")

        epochs_without_improvement = save_best_model(epoch_val_auroc, epoch, net, optimizer, criterion)

        if epochs_without_improvement >= patience:
            break

    load_model = torch.load(save_path)
    net.load_state_dict(load_model['model_state_dict'])

    return net

def train_and_evaluate(model_name, fold, learning_rate, seed, mode, device, mean_filepath, std_filepath, num_epochs=10, batch_size=8, **kwargs):
    """
    Trains and evaluates a model for a given fold.

    Args:
        model_name (str): Name of the model.
        fold (int): Fold number.
        learning_rate (float): Learning rate for the optimizer.
        seed (int): Random seed for reproducibility.
        mode (str): Mode of operation, either 'gridsearch' or 'final'.
        device (str): Device to run the model on.
        mean_filepath (str): Filepath to the mean values.
        std_filepath (str): Filepath to the standard deviation values.
        num_epochs (int, optional): Number of epochs to train the model. Defaults to 10.
        batch_size (int, optional): Batch size for training. Defaults to 8.
        **kwargs: Additional keyword arguments for the model.

    Returns:
        float or None: If mode is 'gridsearch', returns the AUROC (averaged) on the validation set. 
                       If mode is 'final', returns None.
    """
    assert mode in ['gridsearch', 'final']

    # Load data for the current fold
    train_loader, val_loader, test_loader, pos_weight = create_dataset(mean_filepath, std_filepath, fold, seed=seed, batch_size=batch_size)

    # Initialize model, criterion, optimizer, etc.
    net = get_model_instance(model_name, **kwargs).to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.from_numpy(np.array(pos_weight)))
    optimizer = optim.AdamW(net.parameters(), lr=learning_rate)

    # Train the model
    model_save_path = f'../models/{("optuna" if mode == "gridsearch" else "best")}_model_{model_name}_seed_{seed}_fold_{fold}.pth'
    net = train_model(net, train_loader, val_loader, optimizer, criterion, save_path=model_save_path, device=device, num_epochs=num_epochs)

    if mode == 'gridsearch':
        # Evaluate the model
        val_metrics = evaluate_model(net, val_loader, device, mode='validation')
    
        return val_metrics['AUROC (averaged)']
    
    else:

        # Evaluate the model
        test_metrics = get_thresh_and_evaluate(net, device, val_loader, test_loader)
        
        print(test_metrics)
