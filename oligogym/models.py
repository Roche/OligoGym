import pandas as pd
import pickle
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.optim as optim
import xgboost as xgb
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.gaussian_process import GaussianProcessClassifier, GaussianProcessRegressor
from sklearn.linear_model import (
    ElasticNet,
    Lasso,
    LinearRegression,
    LogisticRegression,
    Ridge,
    RidgeClassifier,
)
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.model_selection import train_test_split
from tabpfn import TabPFNRegressor, TabPFNClassifier
from typing import List
from torchinfo import summary
from typing import Union

torch.set_float32_matmul_precision("high")


class SKLearnModel:
    """
    A class to wrap scikit-learn models for training, prediction, saving, and loading.

    Args:
        model (optional): The model to be assigned to the instance. Default is None.
    """

    def __init__(self, model=None):
        """
        Initializes the instance of the class.
        """

        self.model = model

    def fit(self, X, y):
        """
        Fits the model to the provided data.

        This method reshapes the input data `X` if it has more than two dimensions,
        then fits the model using the reshaped data and the target values `y`.

        Args:
            X (numpy.ndarray): The input data to fit the model. If `X` has more than
            two dimensions, it will be reshaped to a 2D array where the first
            dimension is the number of samples.
            y (numpy.ndarray): The target values corresponding to the input data `X`.

        Returns:
            None
        """

        if len(X.shape) > 2:
            X = X.reshape(X.shape[0], -1)
        self.model.fit(X, y)

    def predict(self, X):
        """
        Predict the output for the given input data.

        This method reshapes the input data if it has more than two dimensions
        and then uses the model to predict the output.

        Args:
            X (numpy.ndarray): Input data to be predicted. It can be of any shape,
                       but if it has more than two dimensions, it will be
                       reshaped to a 2D array where the first dimension is
                       the number of samples.

        Returns:
            numpy.ndarray: Predicted output from the model.
        """

        if len(X.shape) > 2:
            X = X.reshape(X.shape[0], -1)
        return self.model.predict(X)

    def save(self, path):
        """
        Save the model to a file.
        Args:
            path (str): The file path where the model will be saved.
        Raises:
            Exception: If there is an issue with saving the model.
        """
        file = open(path, "wb")
        pickle.dump(self.model, file)
        file.close()

    def load(self, path):
        """
        Load a model from a specified file path.
        Args:
            path (str): The file path to load the model from.
        Raises:
            FileNotFoundError: If the specified file does not exist.
            IOError: If there is an error opening or reading the file.
        """
        file = open(path, "rb")
        self.model = pickle.load(file)
        file.close()


class NearestNeighborsModel(SKLearnModel):
    """
    A Nearest Neighbors model for regression or classification tasks.
    This class initializes a Nearest Neighbor model using scikit-learn's
    KNeighborsRegressor for regression tasks or KNeighborsClassifier for
    classification tasks.

    Args:
        task (str): The type of task, either "regression" or "classification". Defaults to "regression".
        n_neighbors (int): The number of neighbors to use. Defaults to 5.
        **model_kwargs: Additional keyword arguments to pass to the model.

    Raises:
        AssertionError: If the task is not "regression" or "classification".
    """

    def __init__(self, task: str = "regression", n_neighbors: int = 5, **model_kwargs):
        """
        Initializes the model with the specified task and number of neighbors.
        """

        assert task in ["regression", "classification"], "Task undefined"
        self.task = task
        self.n_neighbors = n_neighbors

        if self.task == "regression":
            self.model = KNeighborsRegressor(
                n_neighbors=self.n_neighbors, **model_kwargs
            )
        elif self.task == "classification":
            self.model = KNeighborsClassifier(
                n_neighbors=self.n_neighbors, **model_kwargs
            )


class LinearModel(SKLearnModel):
    """
    A class used to represent a Linear Model for regression or classification tasks.
    This class initializes a Linear Model using scikit-learn's LinearRegression, Lasso,
    Ridge or ElasticNet models for regression tasks, and LogisticRegression for
    classification tasks.

    Args:
        task (str): The type of task to perform. Must be either "regression" or "classification".
                    Defaults to "regression".
        type (str): The type of model to use. For "regression", it can be "standard", "lasso",
                    "ridge", or "elastic_net". For "classification", it can be "standard" or "ridge".
                    Defaults to "standard".
        **model_kwargs: Additional keyword arguments to pass to the model constructor.

    Raises:
        AssertionError: If the task is not "regression" or "classification".
    """

    def __init__(
        self, task: str = "regression", type: str = "standard", **model_kwargs
    ):
        """
        Initializes the model based on the specified task and type.
        """
        assert task in ["regression", "classification"], "Task undefined"
        self.task = task
        self.type = type

        if self.task == "regression":
            if self.type == "standard":
                self.model = LinearRegression(**model_kwargs)
            elif self.type == "lasso":
                self.model = Lasso(**model_kwargs)
            elif self.type == "ridge":
                self.model = Ridge(**model_kwargs)
            elif self.type == "elastic_net":
                self.model = ElasticNet(**model_kwargs)
        elif self.task == "classification":
            if self.type == "standard":
                self.model = LogisticRegression(**model_kwargs)
            elif self.type == "ridge":
                self.model = RidgeClassifier(**model_kwargs)


class RandomForestModel(SKLearnModel):
    """
    A model class that wraps around scikit-learn's RandomForestRegressor and RandomForestClassifier.

    Args:
        task (str): The type of task, either "regression" or "classification". Defaults to "regression".
        n_estimators (int): The number of trees in the forest. Defaults to 100.
        **model_kwargs: Additional keyword arguments to pass to the model.

    Raises:
        AssertionError: If the task is not "regression" or "classification".
    """

    def __init__(
        self,
        task: str = "regression",
        n_estimators: int = 100,
        **model_kwargs,
    ):
        """
        Initializes the model based on the specified task.
        """
        assert task in ["regression", "classification"], "Task undefined"
        self.task = task
        if self.task == "regression":
            self.model = RandomForestRegressor(
                n_estimators=n_estimators, **model_kwargs
            )
        elif self.task == "classification":
            self.model = RandomForestClassifier(
                n_estimators=n_estimators, **model_kwargs
            )


class GaussianProcessModel(SKLearnModel):
    """
    A model that uses the sklearn implementation of Gaussian Processes for regression or classification tasks.

    This class wraps around `GaussianProcessRegressor` and `GaussianProcessClassifier`
    from scikit-learn, allowing for easy switching between regression and classification
    tasks based on the `task` parameter.

    Args:
        task (str): The type of task to perform. Must be either "regression" or "classification".
                    Defaults to "regression".
        **model_kwargs: Additional keyword arguments to pass to the model constructor.

    Raises:
        AssertionError: If the task is not "regression" or "classification".
    """

    def __init__(
        self,
        task: str = "regression",
        **model_kwargs,
    ):
        """
        Initializes the model with the specified task type and model parameters.
        """
        assert task in ["regression", "classification"], "Task undefined"
        self.task = task
        if self.task == "regression":
            self.model = GaussianProcessRegressor(**model_kwargs)
        elif self.task == "classification":
            self.model = GaussianProcessClassifier(**model_kwargs)


class XGBoostModel(SKLearnModel):
    """
    A model wrapper for XGBoost that supports both regression and classification tasks.

    Args:
        task (str): The type of task to perform. Must be either "regression" or "classification". Defaults to "regression".
        n_estimators (int): The number of trees in the ensemble. Defaults to 100.
        max_depth (int): The maximum depth of the trees. Defaults to 3.
        **model_kwargs: Additional keyword arguments to pass to the XGBoost model.

    Raises:
        AssertionError: If the task is not "regression" or "classification".
    """

    def __init__(
        self,
        task: str = "regression",
        n_estimators: int = 100,
        max_depth: int = 3,
        **model_kwargs,
    ):
        """
        Initializes the model with specified parameters.
        """
        assert task in ["regression", "classification"], "Task undefined"
        self.task = task
        if self.task == "regression":
            self.model = xgb.XGBRegressor(
                n_estimators=n_estimators, max_depth=max_depth, **model_kwargs
            )
        elif self.task == "classification":
            self.model = xgb.XGBClassifier(
                n_estimators=n_estimators, max_depth=max_depth, **model_kwargs
            )

class TabPFNModel(SKLearnModel):
    """
    A model class that wraps around TabPFNRegressor and TabPFNClassifier.

    Args:
        task (str): The type of task, either "regression" or "classification". Defaults to "regression".
        **model_kwargs: Additional keyword arguments to pass to the model.

    Raises:
        AssertionError: If the task is not "regression" or "classification".
    """

    def __init__(
        self,
        task: str = "regression",
        **model_kwargs,
    ):
        """
        Initializes the model based on the specified task.
        """
        assert task in ["regression", "classification"], "Task undefined"
        self.task = task
        if self.task == "regression":
            self.model = TabPFNRegressor(
                **model_kwargs
            )
        elif self.task == "classification":
            self.model = TabPFNClassifier(
                **model_kwargs
            )

class LightningModel(pl.LightningModule):
    """
    A super-class for storing common methods for training and evaluating neural networks
    implemented using Pytorch Lightning. This class is inherited by the specific neural
    network models such as CNN, MLP, and GRU.
    """

    def __init__(self):
        super().__init__()
        pass

    def fit(
        self,
        X,
        y,
        X_val=None,
        y_val=None,
        val_split: bool = True,
        max_epochs: int = 100,
        early_stopping: bool = True,
        early_stopping_kwargs: dict = {"patience": 5},
        batch_size: int = 32,
        learning_rate: float = 0.001,
        weight_decay: float = 0.01,
        verbose: bool = True,
        **trainer_kwargs,
    ):
        """
        Trains the model using the provided training data and optional validation data.

        Args:
            X (array-like or pd.DataFrame): Training data features.
            y (array-like or pd.DataFrame): Training data labels.
            X_val (array-like or pd.DataFrame, optional): Validation data features. Defaults to None.
            y_val (array-like or pd.DataFrame, optional): Validation data labels. Defaults to None.
            val_split (bool, optional): Whether to split the training data into training and validation sets. Defaults to True.
            max_epochs (int, optional): Maximum number of training epochs. Defaults to 100.
            early_stopping (bool, optional): Whether to use early stopping. Defaults to True.
            early_stopping_kwargs (dict, optional): Additional arguments for early stopping. Defaults to {"patience": 5}.
            batch_size (int, optional): Batch size for training. Defaults to 32.
            learning_rate (float, optional): Learning rate for the optimizer. Defaults to 0.001.
            weight_decay (float, optional): Weight decay for the optimizer. Defaults to 0.01.
            verbose (bool, optional): Whether to print verbose output. Defaults to True.
            **trainer_kwargs: Additional keyword arguments for the trainer.

        Returns:
            None
        """
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay

        if verbose is False:
            pl._logger.setLevel(0)
        if isinstance(X, pd.DataFrame):
            X = X.values
        if isinstance(X_val, pd.DataFrame):
            X_val = X_val.values

        X = torch.tensor(X, dtype=torch.float32)
        y = torch.tensor(y, dtype=torch.float32)
        if len(X.shape) == 3:
            X = X.transpose(1, 2)
        if X_val is not None:
            X_val = torch.tensor(X_val, dtype=torch.float32)
            y_val = torch.tensor(y_val, dtype=torch.float32)
            if len(X_val.shape) == 3:
                X_val = X_val.transpose(1, 2)

        if X_val is None and val_split is True:
            X_train, X_val, y_train, y_val = train_test_split(
                X, y, test_size=0.2, random_state=42
            )
            train_dataloader = torch.utils.data.DataLoader(
                list(zip(X_train, y_train)), batch_size=batch_size, shuffle=True
            )
            val_dataloader = torch.utils.data.DataLoader(
                list(zip(X_val, y_val)), batch_size=batch_size, shuffle=False
            )
        elif X_val is None and val_split is False:
            train_dataloader = torch.utils.data.DataLoader(
                list(zip(X, y)), batch_size=batch_size, shuffle=True
            )
            val_dataloader = None
        elif X_val is not None:
            train_dataloader = torch.utils.data.DataLoader(
                list(zip(X, y)), batch_size=batch_size, shuffle=True
            )
            val_dataloader = torch.utils.data.DataLoader(
                list(zip(X_val, y_val)), batch_size=batch_size, shuffle=False
            )
        if early_stopping:
            early_stop_callback = EarlyStopping(
                monitor="val_loss", mode="min", **early_stopping_kwargs
            )
            self.trainer = pl.Trainer(
                max_epochs=max_epochs, callbacks=[early_stop_callback], **trainer_kwargs
            )
        else:
            self.trainer = pl.Trainer(max_epochs=max_epochs, **trainer_kwargs)

        self.trainer.fit(self, train_dataloader, val_dataloader)
        pl._logger.setLevel(20)

    def training_step(self, batch, batch_idx):
        """
        Training step of the model.

        Args:
            batch (tuple): Input batch containing x and y tensors.
            batch_idx (int): Index of the current batch.

        Returns:
            torch.Tensor: Loss value.
        """
        x, y = batch
        y_hat = self(x)
        y_hat = torch.squeeze(y_hat)
        loss = self.loss_fun(y_hat, y)

        self.log("train_loss", loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        """
        Validation step of the model.

        Args:
            batch (tuple): Input batch containing x and y tensors.
            batch_idx (int): Index of the current batch.
        """
        x, y = batch
        y_hat = self(x)
        y_hat = torch.squeeze(y_hat)
        loss = self.loss_fun(y_hat, y)
        self.log("val_loss", loss, prog_bar=True)

    def configure_optimizers(self):
        """
        Configure the optimizer for the model.

        Returns:
            torch.optim.Optimizer: Optimizer object.
        """
        optimizer = optim.Adam(
            self.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay
        )
        return optimizer

    def print_summary(self, shape):
        """
        Prints the summary of the model.

        Args:
            shape (tuple): The shape of the input data.
        """
        print(summary(self, shape))

    def predict(self, X):
        """
        Predicts the output for the given input data.

        Args:
            X (numpy.ndarray): The input data.

        Returns:
            numpy.ndarray: The predicted output.
        """
        if isinstance(X, pd.DataFrame):
            X = X.values
        X = torch.tensor(X, dtype=torch.float32)
        if len(X.shape) == 3:
            X = X.transpose(1, 2)
        pred = self(X)
        return pred.detach().cpu().numpy()

    def save(self, path):
        """
        Save the model to a file.

        Args:
            path (str): The path to save the model.
        """
        torch.save(self.state_dict(), path)

    def load(self, path):
        """
        Load the model from a file.

        Args:
            path (str): The path to load the model from.
        """
        self.load_state_dict(torch.load(path))


class CNN(LightningModel):
    """
    Convolutional Neural Network (CNN) model for regression or classification tasks.

    This class implements a simple CNN model using PyTorch Lightning. The model consists of multiple convolutional layers followed by a fully connected layer.
    The activation function and pooling operation can be specified, and the model supports both regression and classification tasks.

    Args:
        input_dim (int): The number of input dimensions.
        hidden_dim (int, optional): The number of hidden dimensions. Defaults to 50.
        output_dim (int, optional): The number of output dimensions. Defaults to 1.
        kernel_size (int, optional): The size of the convolutional kernel. Defaults to 6.
        padding (int, optional): The amount of padding for the convolutional layers. Defaults to 0.
        depth (int, optional): The number of convolutional layers. Defaults to 1.
        pooling_operation (str, optional): The type of pooling operation to use. Must be either "max" or "avg". Defaults to "max".
        activation (str, optional): The activation function to use. Must be either "relu" or "leaky_relu". Defaults to "relu".
        task (str, optional): The type of task. Must be either "regression" or "classification". Defaults to "regression".

    Raises:
        AssertionError: If `task` is not "regression" or "classification".
        AssertionError: If `activation` is not "relu" or "leaky_relu".
        AssertionError: If `pooling_operation` is not "max" or "avg".
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 50,
        output_dim: int = 1,
        kernel_size: int = 6,
        padding: int = 0,
        depth: int = 1,
        pooling_operation: str = "max",
        activation: str = "relu",
        task: str = "regression",
    ):
        """
        Initializes the model with the given parameters.
        """
        super().__init__()
        assert task in ["regression", "classification"], "Task undefined"
        assert activation in ["relu", "leaky_relu"], "Activation undefined"
        assert pooling_operation in ["max", "avg"], "Pooling operation undefined"
        self.task = task
        self.pooling_operation = pooling_operation
        self.activation = activation

        self.conv_layers = nn.ModuleList()
        self.conv_layers.append(
            nn.Conv1d(input_dim, hidden_dim, kernel_size=kernel_size, padding=padding)
        )
        for _ in range(depth - 1):
            self.conv_layers.append(
                nn.Conv1d(
                    hidden_dim, hidden_dim, kernel_size=kernel_size, padding=padding
                )
            )
        self.fc = nn.LazyLinear(output_dim)
        self.loss_fun = (
            nn.MSELoss() if self.task == "regression" else nn.CrossEntropyLoss()
        )

    def forward(self, x):
        """
        Forward pass of the CNN model.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor.

        """
        for conv_layer in self.conv_layers:
            x = conv_layer(x)
            if self.activation == "relu":
                x = nn.functional.relu(x)
            elif self.activation == "leaky_relu":
                x = nn.functional.leaky_relu(x)
            if self.pooling_operation == "max":
                x = torch.max_pool1d(x, kernel_size=2)
            elif self.pooling_operation == "avg":
                x = torch.avg_pool1d(x, kernel_size=2)
        x = torch.flatten(x, start_dim=1)
        x = self.fc(x)
        if self.task == "classification":
            x = nn.functional.softmax(x)
        return x


class MLP(LightningModel):
    """
    Multi-Layer Perceptron (MLP) model for regression and classification tasks.
    This model consists of an input layer, one or more hidden layers, and an output layer.
    It supports dropout, batch normalization, and different activation functions.

    Args:
        input_dim (int): Dimension of the input features.
        output_dim (int): Dimension of the output.
        hidden_dims (List[int], optional): List of integers specifying the number of units in each hidden layer. Defaults to [128].
        dropout (float, optional): Dropout rate for regularization. Defaults to 0.25.
        batch_norm (bool, optional): Whether to use batch normalization. Defaults to False.
        activation (str, optional): Activation function to use ('relu' or 'leaky_relu'). Defaults to "relu".
        task (str, optional): Task type ('regression' or 'classification'). Defaults to "regression".

    Raises:
        AssertionError: If the task is not 'regression' or 'classification'.
        AssertionError: If the activation function is not 'relu' or 'leaky_relu'.
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int = 1,
        hidden_dims: List[int] = [128],
        dropout: float = 0.25,
        batch_norm: bool = False,
        activation: str = "relu",
        task: str = "regression",
    ):
        """
        Initializes the MLP model.
        """
        super().__init__()
        assert task in ["regression", "classification"], "Task undefined"
        assert activation in ["relu", "leaky_relu"], "Activation undefined"
        self.task = task
        self.batch_norm = batch_norm
        self.activation = activation

        self.dropout = nn.Dropout(p=dropout)
        self.fc1 = nn.Linear(input_dim, hidden_dims[0])
        self.fc1_batchnorm = nn.BatchNorm1d(hidden_dims[0]) if self.batch_norm else None
        self.hidden_layers = nn.ModuleList(
            [
                nn.Linear(hidden_dims[i], hidden_dims[i + 1])
                for i in range(len(hidden_dims) - 1)
            ]
        )
        self.batch_norm_layers = (
            nn.ModuleList(
                [nn.BatchNorm1d(hidden_dim) for hidden_dim in hidden_dims[1:]]
            )
            if self.batch_norm
            else None
        )
        self.fc2 = nn.Linear(hidden_dims[-1], output_dim)
        self.loss_fun = (
            nn.MSELoss() if self.task == "regression" else nn.CrossEntropyLoss()
        )

    def forward(self, x):
        """
        Forward pass of the MLP.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor.

        """
        if len(x.shape) == 3:
            x = torch.flatten(x, start_dim=1)
        x = self.fc1(x)
        if self.activation == "relu":
            x = nn.functional.relu(x)
        elif self.activation == "leaky_relu":
            x = nn.functional.leaky_relu(x)
        if self.batch_norm:
            x = self.fc1_batchnorm(x)
        x = self.dropout(x)
        for i in range(len(self.hidden_layers)):
            x = self.hidden_layers[i](x)
            if self.activation == "relu":
                x = nn.functional.relu(x)
            elif self.activation == "leaky_relu":
                x = nn.functional.leaky_relu(x)
            if self.batch_norm:
                x = self.batch_norm_layers[i](x)
            x = self.dropout(x)
        x = self.fc2(x)
        if self.task == "classification":
            x = nn.functional.softmax(x)
        return x


class GRU(LightningModel):
    """
    Gated Recurrent Unit (GRU) model for regression or classification tasks.
    This model uses a GRU layer followed by a fully connected layer to perform
    either regression or classification. The number of GRU layers and the task
    type can be specified during initialization.

    Args:
        input_dim (int): The number of input dimensions.
        hidden_dim (int): The number of hidden dimensions.
        output_dim (int): The number of output dimensions.
        num_layers (int, optional): The number of GRU layers. Defaults to 1.
        task (str, optional): The task type, either "regression" or "classification". Defaults to "regression".
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int = 1,
        num_layers: int = 1,
        task: str = "regression",
    ):
        """
        Initialize the GRU model.
        """
        super().__init__()
        assert task in ["regression", "classification"], "Task undefined"
        self.task = task

        self.gru = nn.GRU(input_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.LazyLinear(output_dim)
        self.loss_fun = (
            nn.MSELoss() if self.task == "regression" else nn.CrossEntropyLoss()
        )

    def forward(self, x):
        """
        Forward pass of the GRU model.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor.
        """
        x = x.transpose(1, 2)
        x, _ = self.gru(x)
        x = x[:, -1, :]
        x = self.fc(x)
        if self.task == "classification":
            x = nn.functional.softmax(x, dim=1)
        return x


class CausalCNN(LightningModel):
    """
    Causal Convolutional Neural Network (CausalCNN) for sequence data.
    This model is designed to handle sequence data using causal convolutions,
    which ensure that the model does not violate the sequential order of the data.
    It supports both regression and classification tasks.

    Args:
        input_dim (int): The number of input dimensions.
        hidden_dim (int, optional): The number of hidden dimensions. Defaults to 50.
        output_dim (int, optional): The number of output dimensions. Defaults to 1.
        kernel_size (int, optional): The size of the convolutional kernel. Defaults to 6.
        depth (int, optional): The number of convolutional layers. Defaults to 1.
        dilation_rates (Union[int, list], optional): The dilation rates for the convolutional layers.
            Can be a single integer or a list of integers. Defaults to 1.
        pooling_operation (str, optional): The pooling operation to use. Must be either "max" or "avg". Defaults to "max".
        activation (str, optional): The activation function to use. Must be either "relu" or "leaky_relu". Defaults to "relu".
        task (str, optional): The task type. Must be either "regression" or "classification". Defaults to "regression".

    Raises:
        AssertionError: If `task` is not "regression" or "classification".
        AssertionError: If `activation` is not "relu" or "leaky_relu".
        AssertionError: If `pooling_operation` is not "max" or "avg".
        AssertionError: If the length of `dilation_rates` does not match `depth` when `dilation_rates` is a list.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 50,
        output_dim: int = 1,
        kernel_size: int = 6,
        depth: int = 1,
        dilation_rates: Union[int, list] = 1,
        pooling_operation: str = "max",
        activation: str = "relu",
        task: str = "regression",
    ):
        """
        Initializes the model with the given parameters.
        """
        super().__init__()
        assert task in ["regression", "classification"], "Task undefined"
        assert activation in ["relu", "leaky_relu"], "Activation undefined"
        assert pooling_operation in ["max", "avg"], "Pooling operation undefined"

        if isinstance(dilation_rates, int):
            dilation_rates = [dilation_rates] * depth
        elif len(dilation_rates) == 1:
            dilation_rates = dilation_rates * depth
        assert len(dilation_rates) == depth, "Length of dilation_rates must match depth"

        self.task = task
        self.pooling_operation = pooling_operation
        self.activation = activation

        self.conv_layers = nn.ModuleList()
        self.conv_layers.append(
            nn.Conv1d(
                input_dim,
                hidden_dim,
                kernel_size=kernel_size,
                padding=(kernel_size - 1) * dilation_rates[0],
                dilation=dilation_rates[0],
            )
        )
        for i in range(1, depth):
            self.conv_layers.append(
                nn.Conv1d(
                    hidden_dim,
                    hidden_dim,
                    kernel_size=kernel_size,
                    padding=(kernel_size - 1) * dilation_rates[i],
                    dilation=dilation_rates[i],
                )
            )
        self.fc = nn.LazyLinear(output_dim)
        self.loss_fun = (
            nn.MSELoss() if self.task == "regression" else nn.CrossEntropyLoss()
        )

    def forward(self, x):
        """
        Forward pass of the Causal CNN model.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor.
        """
        for conv_layer in self.conv_layers:
            x = conv_layer(x)
            x = x[:, :, : -conv_layer.padding[0]]
            if self.activation == "relu":
                x = nn.functional.relu(x)
            elif self.activation == "leaky_relu":
                x = nn.functional.leaky_relu(x)

        if self.pooling_operation == "max":
            x = nn.functional.adaptive_max_pool1d(x, output_size=1)
        elif self.pooling_operation == "avg":
            x = nn.functional.adaptive_avg_pool1d(x, output_size=1)

        x = torch.flatten(x, start_dim=1)
        x = self.fc(x)
        if self.task == "classification":
            x = nn.functional.softmax(x, dim=1)
        return x
