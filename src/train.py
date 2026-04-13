from copy import deepcopy

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import DataLoader
import logging
import copy
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score
from collections.abc import Sequence

from src.utils import (calculate_sensitivity_specificity, log_optuna_metrics, save_models,
                       log_metrics_stats, find_optimal_threshold)

import torch
import numpy as np
from sklearn.metrics import accuracy_score, f1_score


# noinspection t
def run_epoch(model, loader, criterion, optimizer, device, is_training: bool):
    """
    Optimized epoch runner for multi-class classification.
    """
    model.train() if is_training else model.eval()

    total_loss = 0.0
    all_preds = []
    all_labels = []

    # Use torch.set_grad_enabled to handle inference vs training context globally
    with torch.set_grad_enabled(is_training):
        for inputs, labels in loader:
            # 1. Device Management & Data Prep
            if isinstance(inputs, (list, tuple)):
                inputs = [i.to(device) for i in inputs]
            else:
                inputs = inputs.to(device)

            labels = labels.to(device).long()

            # 2. Forward Pass
            outputs = model(inputs)

            # Ensure 2D shape [Batch, Classes]
            if outputs.dim() == 1:
                outputs = outputs.unsqueeze(0)

            # Handle specific BCE logic if necessary (though CrossEntropy is standard for multi-class)
            if isinstance(criterion, torch.nn.BCEWithLogitsLoss):
                loss = criterion(outputs.squeeze(1), labels.float())
            else:
                loss = criterion(outputs, labels)

            # 3. Backward Pass
            if is_training:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            # 4. Collection (Detached from graph)
            total_loss += loss.item()
            # Check if the output is binary (shape [Batch, 1]) or multi-class (shape [Batch, Classes])
            if outputs.shape[1] == 1:
                # Binary: Logits > 0 means Probability > 0.5
                preds = (outputs > 0.0).long().squeeze(-1)
            else:
                # Multi-class: Take the highest logit
                preds = outputs.argmax(dim=1)

            all_preds.append(preds.detach().cpu())
            all_labels.append(labels.detach().cpu())

    # 5. Metrics Calculation
    y_pred = torch.cat(all_preds).numpy()
    y_true = torch.cat(all_labels).numpy()

    avg_loss = total_loss / len(loader)
    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average="macro")

    metrics = {"loss": avg_loss, "accuracy": acc, "f1": f1}
    results = {"labels": y_true, "predictions": y_pred}

    return avg_loss, metrics, results

def train_one_fold(model, train_loader, val_loader, criterion, optimizer, device, num_epochs):
    """
    Trains the model for one fold and returns the best model state based on validation AUC.
    """
    best_val_auc = 0
    best_model_state = None
    best_val_metrics = {}
    best_threshold = 0.5
    best_val_ys = None  # Store the full validation predictions dict

    for epoch in range(num_epochs):
        train_loss, train_metrics, _ = run_epoch(model, train_loader, criterion, optimizer, device, is_training=True)
        val_loss, val_metrics, val_ys = run_epoch(model, val_loader, criterion, optimizer, device, is_training=False)

        if epoch % 5 == 0:
            logging.info(
                f"Epoch {epoch + 1}/{num_epochs} | "
                f"Train Loss: {train_loss:.4f}, Acc: {train_metrics['accuracy']:.4f}, f1: {train_metrics['f1']:.4f} | "
                f"Val Loss: {val_loss:.4f}, Acc: {val_metrics['accuracy']:.4f}, f1: {val_metrics['f1']:.4f}"
            )

        if val_metrics['f1'] > best_val_auc:
            best_val_auc = val_metrics['f1']
            best_val_metrics = val_metrics
            best_model_state = copy.deepcopy(model.state_dict())

    return best_model_state, best_val_metrics

def train_and_evaluate_model(
    trial, dataloaders, test_df_dict, exclude_columns,
    num_epochs=30, learning_rate=0.001, batch_size=32,
    model_cls=None, model_kwargs=None,
    dataset_cls=None, dataset_kwargs=None
):
    """
    Trains and evaluates the model using cross-validation.
    model_cls: class of the model to instantiate.
    model_kwargs: dict of kwargs to pass to the model constructor.
    dataset_cls: class of the dataset to instantiate for test set.
    dataset_kwargs: dict of kwargs to pass to the dataset constructor for test set.
    """
    if model_cls is None:
        from src.models import SimpleNN
        model_cls = SimpleNN
    if dataset_cls is None:
        from src.dataset import ClinicalDataset
        dataset_cls = ClinicalDataset
    if dataset_kwargs is None:
        dataset_kwargs = {"columns_to_drop": exclude_columns}

    weight_decay = trial.suggest_float("weight_decay", 5e-3, 1e-1, log=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if model_kwargs.get("out_dim", 1) > 1:
        criterion = nn.CrossEntropyLoss()
    else:
        criterion = nn.BCEWithLogitsLoss()

    test_metrics = {"loss": [], "accuracy": [], "f1": []}
    external_test_metrics = deepcopy(test_metrics)

    val_metrics_folds = {"loss": [], "accuracy": [], "f1": []}
    outputs_and_predictions = {"labels": [], "predictions": [], "external_labels": [], "external_predictions": []}
    models = []

    for fold, loaders in dataloaders.items():
        logging.info(f"Training Fold {fold + 1}")
        train_loader, val_loader = loaders['train'], loaders['val']

        # Instantiate a new model and optimizer for each fold (reset parameters)
        model = model_cls(**model_kwargs).to(device)
        optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

        best_model_state, val_metrics = train_one_fold(
            model, train_loader, val_loader, criterion, optimizer, device, num_epochs
        )

        val_metrics_folds["loss"].append(val_metrics["loss"])
        val_metrics_folds["accuracy"].append(val_metrics["accuracy"])
        val_metrics_folds["f1"].append(val_metrics["f1"])

        if best_model_state is not None:
            model.load_state_dict(best_model_state)

        test_metrics, test_ys = evaluate_on_test_set(
            model, test_df_dict, criterion, device, batch_size, test_metrics,
            dataset_cls=dataset_cls, dataset_kwargs=dataset_kwargs
        )

        if test_metrics["accuracy"][fold] > 0.5 and test_metrics["f1"][fold] > 0.55:
            models.append(model.state_dict())

        outputs_and_predictions["labels"].append(test_ys["labels"])
        outputs_and_predictions["predictions"].append(test_ys["predictions"])


        np.save(f"predictions/{trial.user_attrs['modality']}/outputs_and_predictions_{trial.number}.npy",
                outputs_and_predictions)


    # Test set stats
    mean_test_metrics = {metric: np.mean(values) for metric, values in test_metrics.items()}
    std_test_metrics = {metric: np.std(values) for metric, values in test_metrics.items()}
    # Validation set stats
    mean_val_metrics = {metric: np.mean(values) for metric, values in val_metrics_folds.items()}

    # log to optuna
    log_optuna_metrics(trial, mean_val_metrics)
    log_optuna_metrics(trial, mean_test_metrics, is_test=True, is_external=False)

    # Log to log file
    log_metrics_stats(trial, mean_test_metrics, std_test_metrics)

    # Model saving, Duh!!
    # save_models(models, trial, mean_test_metrics)



    return mean_val_metrics


def evaluate_on_test_set(model, test_df, criterion, device, batch_size, test_metrics,
                         dataset_cls, dataset_kwargs):

    # Create test loader
    test_loader = DataLoader(
        dataset_cls(test_df, **dataset_kwargs),
        batch_size=batch_size,
        shuffle=False
    )

    model.eval()

    # Run evaluation epoch (multi-class)
    test_loss, test_metrics_fold, test_ys = run_epoch(
        model, test_loader, criterion, None, device, is_training=False
    )

    # Extract predictions directly (already argmax outputs)
    y_true = test_ys["labels"]
    y_pred = test_ys["predictions"]

    # Metrics
    accuracy = accuracy_score(y_true, y_pred)
    f1 = test_metrics_fold["f1"]

    # Store metrics
    test_metrics["loss"].append(test_loss)
    test_metrics["accuracy"].append(accuracy)
    test_metrics["f1"].append(f1)

    return test_metrics, test_ys
