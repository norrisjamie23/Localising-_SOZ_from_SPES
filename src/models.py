import torch
import torch.nn as nn
from multi_scale_ori import MSResNet
from torcheeg.transforms import RandomNoise
import random

class SPES_ResNet(nn.Module):
    """
    A modified ResNet model for iEEG data processing, incorporating multi-scale features.
    The model applies random noise to the input during training and selects random channels
    from the input data before passing it through the MSResNet architecture.
    """
    def __init__(self, input_channels=40, num_classes=1, divergent=True, **kwargs):
        super(SPES_ResNet, self).__init__()
        self.divergent = divergent
        self.input_channels = input_channels
        self.msresnets = MSResNet(input_channel=input_channels, num_classes=num_classes, **kwargs)  # Multi-Scale ResNet
        self.noise = RandomNoise(std=0.1)  # Random noise transformer
        self.fc = nn.Linear(768, num_classes)  # Final fully connected layer

    def forward(self, x):
        """
        Forward pass of the model.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, num_channels, height, width).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, num_classes).
        """
        if self.divergent:
            x = x[0]
        else:
            x = x[1]

        # Extracting distances from the input data.
        distances = x[:, 0, :, 0]

        # Apply noise to the non-distance channels during training.
        if self.training:
            x[:, :, :, 1:] = self.noise(eeg=x[:, :, :, 1:])['eeg']
        
        all_x = []

        # Process each sample in the batch.
        for single_sample, distance in zip(x, distances):
            valid_rows = torch.where(distance != 0)[0]
            p = torch.ones(len(valid_rows), device=x.device) / len(valid_rows)
            idx = p.multinomial(num_samples=self.input_channels, replacement=len(valid_rows) < self.input_channels)
            random_channels = valid_rows[idx]
            random_channels = random_channels.sort()[0]
            
            all_x.append(single_sample[0, random_channels, 1:])
        
        # Stack processed samples and pass them through the MSResNet and the final layer.
        x = torch.stack(all_x, dim=0)
        x = self.msresnets(x)
        x = self.fc(x)

        return x


class SPESResponseEncoder(nn.Module):
    """
    A neural network model for classifying responses to Single Pulse Electrical Stimulation (SPES).
    The full model incorporates both convolutional and MLP embeddings, with a transformer encoder for the final classification.
    """
    def __init__(self, mean: bool, std: bool, conv_embedding: bool = True, mlp_embedding: bool = True, 
                    dropout_rate=0.5, num_layers=2, embedding_dim=64, random_channels=None):
        """
        Initialize the SPESResponseEncoder class.

        Args:
            mean (bool): Flag indicating whether to include mean in embedding.
            std (bool): Flag indicating whether to include standard deviation in embedding.
            conv_embedding (bool, optional): Flag indicating whether to use convolutional embedding. Defaults to True.
            mlp_embedding (bool, optional): Flag indicating whether to use MLP embedding. Defaults to True.
            dropout_rate (float, optional): Dropout rate. Defaults to 0.5.
            num_layers (int, optional): Number of transformer encoder layers. Defaults to 2.
            embedding_dim (int, optional): Dimension of the embedding. Defaults to 64.
            random_channels (None, optional): Random channels. Defaults to None.
        """
        super(SPESResponseEncoder, self).__init__()

        assert mean or std, "Either mean or std (or both) must be True for embedding."

        self.mean = mean
        self.std = std
        self.conv_embedding = conv_embedding
        self.mlp_embedding = mlp_embedding
        self.random_channels = random_channels

        if conv_embedding:
            input_channels = self.mean + self.std
            self.msresnet = MSResNet(input_channel=input_channels, num_classes=1)
            embedding_in = 768 + (self.mean + self.std) * 155 * mlp_embedding
        else:
            embedding_in = (self.mean + self.std) * 509

        self.patch_to_embedding = nn.Linear(embedding_in, embedding_dim)

        self.dropout = nn.Dropout(dropout_rate)
        self.class_token = nn.Parameter(nn.init.xavier_normal_(torch.empty(1, 1, embedding_dim)))

        encoder_layer = nn.TransformerEncoderLayer(d_model=embedding_dim, nhead=embedding_dim // 8, dim_feedforward=embedding_dim * 2, dropout=dropout_rate, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.noise = RandomNoise(std=0.1)

    def forward(self, x):
        """
        Forward pass of the model.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, channels, height, width).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, hidden_size).
        """
        if self.training:
            # Applying noise and zeroing random channels during training
            x = self.apply_noise_and_zero_channels(x)
        
        if self.random_channels:
            distances = x[:, 0, :, 0]

            all_x = []

            # Process each sample in the batch.
            for single_sample, distance in zip(x, distances):
                valid_rows = torch.where(distance != 0)[0]

                if len(valid_rows) < self.random_channels:
                    all_x.append(single_sample[:, :self.random_channels])

                else:
                    p = torch.ones(len(valid_rows), device=x.device) / len(valid_rows)
                    idx = p.multinomial(num_samples=self.random_channels, replacement=False)
                    random_channels = valid_rows[idx]
                    random_channels = random_channels.sort()[0]
                    
                    all_x.append(single_sample[:, random_channels])
        
            # Stack processed samples and pass them through the MSResNet and the final layer.
            x = torch.stack(all_x, dim=0)

        distances = x[:, 0, :, 0]
        key_padding_mask = self.create_key_padding_mask(distances)

        all_output = self.prepare_channels(x)

        x = self.dropout(self.patch_to_embedding(all_output))

        weight = self.class_token.repeat(x.shape[0], 1, 1)
        x = torch.cat((weight, x), dim=1)

        x = self.transformer_encoder(x, src_key_padding_mask=key_padding_mask)

        return x[:, 0]

    def apply_noise_and_zero_channels(self, x):
        """
        Applies noise to EEG data and zeros out random channels.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, num_channels, num_samples, num_features).

        Returns:
            torch.Tensor: Output tensor with noise applied and random channels zeroed out.
        """

        # Implementation of noise application and zeroing random channels
        non_zero_indices = torch.nonzero(x[:, 0, :, 0].sum(axis=0), as_tuple=False)

        # Step 1: Uniformly sample a number from 0 to the length of non_zero_indices
        sample_size = random.randint(0, len(non_zero_indices) // 2)

        # Step 2: Select a random sample of this number from non_zero_indices without replacement
        random_indices = torch.randperm(len(non_zero_indices))[:sample_size]
        random_sample = non_zero_indices[random_indices]

        # Set these to zero
        x[:, :, random_sample] = 0

        x[:, :, :, 1:] = self.noise(eeg=x[:, :, :, 1:])['eeg']

        return x

    def create_key_padding_mask(self, distances):
        """
        Creates a key padding mask based on the distances tensor.

        Args:
            distances (torch.Tensor): Tensor containing the distances.

        Returns:
            torch.Tensor: Key padding mask tensor.
        """

        # Distance == 0 means padding channel
        key_padding_mask = distances == 0

        # Create a tensor of False values with shape (n, 1)
        false_column = torch.zeros(distances.size(0), 1, dtype=torch.bool, device=distances.device)

        # Concatenate the false_column tensor with the distances tensor
        key_padding_mask = torch.cat([false_column, key_padding_mask], dim=1)

        return key_padding_mask

    def prepare_channels(self, x):
        """
        Prepares each channel prior to embedding.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, modes, chans, timesteps).

        Returns:
            torch.Tensor: Output tensor after channel preparation.
        """
        if self.conv_embedding:
            if self.mean:
                if self.std:
                    conv_input = x[:, :, :, 1:]
                else:
                    conv_input = x[:, :1, :, 1:]
            else:
                conv_input = x[:, 1:, :, 1:]

            batch_size, modes, chans, timesteps = conv_input.shape
            conv_input.swapaxes(1, 2)
            conv_input = conv_input.reshape(-1, modes, timesteps)

            late_output = self.msresnet(conv_input)

            late_output = late_output.reshape(batch_size, chans, -1)

            if self.mlp_embedding:
                if self.mean:
                    if self.std:
                        all_output = torch.cat([x[:, 0, :, :155], x[:, 1, :, :155], late_output], dim=-1)
                    else:
                        all_output = torch.cat([x[:, 0, :, :155], late_output], dim=-1)
                else:
                    all_output = torch.cat([x[:, 1, :, :155], late_output], dim=-1)
            else:
                all_output = late_output
        elif self.mlp_embedding:
            if self.mean:
                if self.std:
                    all_output = torch.cat([x[:, 0], x[:, 1]], dim=-1)
                else:
                    all_output = x[:, 0]
            else:
                all_output = x[:, 1]

        return all_output


class SPES_Transformer(nn.Module):
    """
    A neural network model integrating one or more SPESResponseEncoder instances and classifying the concatenated response.
    This model can handle different types of input data, specified as 'convergent' or 'divergent'.
    
    Args:
        num_classes (int): The number of output classes for classification.
        net_configs (list): A list of dictionaries specifying the configuration for each SPESResponseEncoder instance.
        dropout_rate (float, optional): The dropout rate to be applied. Defaults to 0.5.
        **kwargs: Additional keyword arguments for the SPESResponseEncoder instances.
    """
    def __init__(self, num_classes: int, net_configs: list, dropout_rate=0.5, **kwargs):
        super(SPES_Transformer, self).__init__()

        self.net_configs = net_configs

        # Initialize SPESResponseEncoder instances based on net_configs
        self.eegnets = nn.ModuleList([
            SPESResponseEncoder(mean=net_config['mean'], std=net_config['std'], dropout_rate=dropout_rate, **kwargs) 
            for net_config in net_configs
        ])

        total_feature_size = kwargs['embedding_dim'] * len(net_configs) if 'embedding_dim' in kwargs else 64 * len(net_configs)

        self.fc = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(total_feature_size, num_classes)
        )

        nn.init.xavier_uniform_(self.fc[1].weight)
        if self.fc[1].bias is not None:
            nn.init.zeros_(self.fc[1].bias)

    def forward(self, inputs):
        """
        Forward pass of the model.

        Args:
            inputs (list): List of two input tensors, one for convergent data and one for divergent data.

        Returns:
            torch.Tensor: Output tensor after passing through the model.
        """
        # Validate inputs
        if len(inputs) != 2:
            raise ValueError("Expected two input tensors in the list: one for convergent and one for divergent data.")

        processed_inputs = []

        # Process each input through its corresponding EEGNetBackbone
        for net_config, eegnet in zip(self.net_configs, self.eegnets):
            if net_config['type'] == 'convergent':
                input_data = inputs[1]
            elif net_config['type'] == 'divergent':
                input_data = inputs[0]
            else:
                raise ValueError(f"Invalid type '{net_config['type']}' in net_configs; must be 'convergent' or 'divergent'.")
            
            processed_inputs.append(eegnet(input_data))

        # Concatenate the outputs from all EEGNetBackbone instances
        x = torch.cat(processed_inputs, dim=1)

        # Pass the concatenated output through the fully connected layer
        x = self.fc(x)
        return x


def get_model_instance(model_name, **kwargs):
    """
    Returns an instance of a specific model based on the given model_name.

    Args:
        model_name (str): The name of the model to instantiate.
        **kwargs: Additional keyword arguments to be passed to the model constructor.

    Returns:
        An instance of the specified model.

    Raises:
        ValueError: If an invalid model_name is provided.
    """
    if model_name == 'Transformer (all)':
        return SPES_Transformer(num_classes=1, net_configs=[{'type': 'convergent', 'mean': True, 'std': True}], **kwargs)
    elif model_name == 'Transformer (base)':
        return SPES_Transformer(num_classes=1, net_configs=[{'type': 'convergent', 'mean': True, 'std': False}], mlp_embedding=False, **kwargs)
    elif model_name == 'CNN (divergent)':
        return SPES_ResNet(num_classes=1, **kwargs)
    elif model_name == 'CNN (convergent)':
        return SPES_ResNet(num_classes=1, divergent=False, **kwargs)
    else:
        raise ValueError(f"Invalid model_name: {model_name}")


def load_model_from_path(model_name, model_path, device, **hyperparams):
    """
    Load a trained model from a given file path.

    Args:
        model_name (str): The name of the model to be loaded.
        model_path (str): The file path of the saved model.
        device (torch.device): The device to load the model onto.
        **hyperparams: Additional hyperparameters to be passed to the model.

    Returns:
        torch.nn.Module: The loaded model.
    """
    model = get_model_instance(model_name, **hyperparams).to(device)

    loaded_model = torch.load(model_path)
    model.load_state_dict(loaded_model['model_state_dict'])

    return model


class SaveBestModel:
    """
    Class to save the best model while training. If the current epoch's 
    validation metric is better than the best so far, the model state is saved.
    Stops saving if there's no improvement in the validation metric for a specified
    number of epochs (patience).

    Args:
        patience (int): Number of epochs without improvement before stopping saving.
        mode (str): Mode of improvement. 'max' for higher metric, 'min' for lower metric.
        save_path (str): Path to save the best model.

    Attributes:
        best_valid_metric (float): Best validation metric achieved so far.
        patience (int): Number of epochs without improvement before stopping saving.
        epochs_without_improvement (int): Counter to track epochs without improvement.
        mode (str): Mode of improvement. 'max' for higher metric, 'min' for lower metric.
        save_path (str): Path to save the best model.

    Methods:
        __call__(self, current_valid_metric, epoch, model, optimizer, criterion):
            Saves the best model if there is improvement in the validation metric.
            Updates the best validation metric and epochs without improvement counter.
            Returns the updated epochs without improvement counter.
    """
    def __init__(self, patience=10, mode='max', save_path='../models/best_model.pth'):
        self.best_valid_metric = float('-inf') if mode=='max' else float('inf')
        self.patience = patience
        self.epochs_without_improvement = 0  # Counter to track epochs without improvement
        self.mode = mode # Save model if validation metric is higher than the best so far
        self.save_path = save_path
        
    def __call__(self, current_valid_metric, epoch, model, optimizer, criterion):
        improvement = current_valid_metric > self.best_valid_metric if self.mode == 'max' else current_valid_metric < self.best_valid_metric
        if improvement:
            self.best_valid_metric = current_valid_metric
            self.epochs_without_improvement = 0  # Reset counter on improvement
            print(f"\nBest validation metric: {self.best_valid_metric}")
            print(f"\nSaving best model for epoch: {epoch}\n")
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': criterion,
                }, self.save_path)
        else:
            self.epochs_without_improvement += 1  # Increment counter if no improvement

        # Return patience counter to track epochs without improvement
        return self.epochs_without_improvement
