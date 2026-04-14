
from monai.metrics import DiceMetric, SurfaceDiceMetric

import optuna
import torch
import torchio as tio
from monai.losses import DiceCELoss
import numpy as np
import logging
import gc


def train_epoch(model, dataloader, criterion, optimizer, device):
    """Handles a single training epoch."""
    model.train()
    train_losses = []

    for batch in dataloader:
        inputs = batch['t1c'][tio.DATA].to(device)
        targets = batch['mask'][tio.DATA].to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)

        # Catch NaNs immediately
        if torch.isnan(loss):
            logging.error("NaN Loss detected during training.")
            return float('nan')

        loss.backward()
        optimizer.step()
        train_losses.append(loss.item())

    return np.mean(train_losses)


def validate_epoch(model, dataloader, device):
    """Handles a single validation epoch and manual Dice calculation."""
    model.eval()
    epoch_dices = []
    positive_preds_count = 0
    total_patches = 0

    with torch.no_grad():
        for batch in dataloader:
            inputs = batch['t1c'][tio.DATA].to(device)
            targets = batch['mask'][tio.DATA].to(device)
            outputs = model(inputs)

            probs = torch.sigmoid(outputs)
            preds = (probs > 0.5).float()

            # Stats for debugging
            total_patches += inputs.size(0)
            if preds.sum() > 0:
                positive_preds_count += 1

            # Smart Dice Calculation [B, 1, D, H, W] -> [B]
            intersection = (preds * targets).flatten(start_dim=1).sum(dim=1)
            union = preds.flatten(start_dim=1).sum(dim=1) + targets.flatten(start_dim=1).sum(dim=1)

            batch_dices = torch.zeros(inputs.size(0), device=device)

            # Case A: Both empty (Correct rejection) -> Score 1.0
            empty_mask = (union == 0)
            batch_dices[empty_mask] = 1.0

            # Case B: Not empty -> Standard Dice formula
            non_empty_mask = ~empty_mask
            if non_empty_mask.any():
                batch_dices[non_empty_mask] = (2. * intersection[non_empty_mask]) / (union[non_empty_mask] + 1e-5)

            epoch_dices.extend(batch_dices.cpu().tolist())

    mean_dice = np.mean(epoch_dices) if epoch_dices else 0.0
    return mean_dice, positive_preds_count, total_patches


# noinspection t
def train_one_fold_seg(model, train_loader, val_loader, criterion, optimizer, device, num_epochs, patience=15, trial=None, fold_idx=0):
    """Main loop orchestrating training, validation, and early stopping."""
    best_dice = 0.0
    best_model_state = model.state_dict()
    epochs_no_improve = 0

    criterion = DiceCELoss(sigmoid=True, squared_pred=False, reduction='mean')

    for epoch in range(num_epochs):

        # 1. Train
        avg_train_loss = train_epoch(model, train_loader, criterion, optimizer, device)

        # Abort if model exploded
        if np.isnan(avg_train_loss):
            logging.error(f"Stopping training early at Epoch {epoch + 1} due to NaN loss.")
            if trial is not None:
                raise optuna.TrialPruned() # Prune immediately on NaN
            return best_model_state, 0.0

        # 2. Validate
        mean_dice, pos_preds_count, total_patches = validate_epoch(model, val_loader, device)

        # 3. Logging
        pred_ratio = (pos_preds_count / total_patches * 100) if total_patches else 0
        logging.info(
            f"Epoch {epoch + 1}: Loss {avg_train_loss:.4f} | Val Dice: {mean_dice:.4f} | Non-Empty Preds: {pred_ratio:.1f}%"
        )

        # OPTUNA PRUNING LOGIC
        if trial is not None:
            step = (fold_idx * num_epochs) + epoch
            trial.report(mean_dice, step)
            if trial.should_prune():
                logging.info(f"Trial pruned by Optuna at Fold {fold_idx}, Epoch {epoch + 1}")
                raise optuna.TrialPruned()

        # 4. Early Stopping and Checkpointing
        if mean_dice > best_dice:
            best_dice = mean_dice
            best_model_state = model.state_dict()
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        if epochs_no_improve >= patience:
            logging.info(f"Early stopping triggered at epoch {epoch + 1}")
            break

    return best_model_state, best_dice


def masking_function(x):
    """Return a boolean mask where values > 0 are True."""
    return x > 0


# noinspection t
def test_inference(model, test_subjects, device, patch_size=(64, 64, 64), patch_overlap=(16, 16, 16), batch_size=4):
    dice_metric = DiceMetric(include_background=True, reduction="mean")
    surface_dice_metric = SurfaceDiceMetric(class_thresholds=[0.5], include_background=True)
    # 2. Define the exact same preprocessing used in training
    pad = 32
    pre_processing = tio.Compose([
        tio.Pad((pad, pad, pad)),
        tio.ZNormalization(masking_method=masking_function),
    ])

    dice_scores = []
    surface_dice_scores = []

    for i, subject in enumerate(test_subjects):
        subject = pre_processing(subject)
        grid_sampler = tio.inference.GridSampler(subject, patch_size, patch_overlap)
        patch_loader = torch.utils.data.DataLoader(grid_sampler, batch_size=batch_size)
        aggregator = tio.inference.GridAggregator(grid_sampler)

        model.eval()
        with torch.no_grad():
            for patches_batch in patch_loader:
                inputs = patches_batch['t1c'][tio.DATA].to(device)
                locations = patches_batch[tio.LOCATION]
                outputs = model(inputs)
                outputs = torch.sigmoid(outputs)  # Fix: Logits to Probs
                aggregator.add_batch(outputs, locations)

        pred_volume = aggregator.get_output_tensor()
        gt_volume = subject['mask'][tio.DATA]

        orig_d = pred_volume.shape[-3] - 2 * pad
        orig_h = pred_volume.shape[-2] - 2 * pad
        orig_w = pred_volume.shape[-1] - 2 * pad
        pred_volume = pred_volume[..., pad:pad + orig_d, pad:pad + orig_h, pad:pad + orig_w]
        gt_volume = gt_volume[..., pad:pad + orig_d, pad:pad + orig_h, pad:pad + orig_w]

        # Fix: Add Batch Dimension [1, 1, D, H, W]
        if pred_volume.ndim == 4:
            pred_volume = pred_volume.unsqueeze(0)
        if gt_volume.ndim == 4:
            gt_volume = gt_volume.unsqueeze(0)

        pred_bin = (pred_volume > 0.5).float()

        dice = dice_metric(pred_bin, gt_volume)
        dice_scores.append(dice.item())

        try:
            if pred_bin.sum() > 0 and gt_volume.sum() > 0:
                surf_dice = surface_dice_metric(pred_bin, gt_volume)
                surface_dice_scores.append(surf_dice.item())
            else:
                surface_dice_scores.append(0.0)
        except Exception as e:
            # LOGGING: Warning for metric failures
            logging.warning(f"Surface Dice failed for subject {i}: {e}")
            surface_dice_scores.append(0.0)

        # --- NEW CODE: FORCE MEMORY CLEANUP ---
        # Delete the massive full-volume variables
        del pred_volume, gt_volume, pred_bin

        # If running on GPU, clear the PyTorch cache
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # Force Python to immediately release the unreferenced RAM
        gc.collect()

    return np.mean(dice_scores), np.mean(surface_dice_scores)


def run_patch_segmentation_cv(model_cls, fold_dataloaders, test_subjects, device,
                              num_epochs=50, lr=1e-3, wd=1e-5, patience=10, trial=None):
    val_results = []
    test_dice_scores = []
    test_surface_scores = []

    logging.info("=== Starting Cross-Validation Patch Segmentation ===")

    # Enumerate to track fold_idx for Optuna steps
    for fold_idx, (fold_id, loaders) in enumerate(fold_dataloaders.items()):
        logging.info(f"--- Starting Fold {fold_id} ---")

        model = model_cls().to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
        criterion = torch.nn.BCEWithLogitsLoss()

        best_model_state, best_val_dice = train_one_fold_seg(
            model, loaders['train'], loaders['val'], criterion, optimizer, device, num_epochs, patience,
            trial=trial, fold_idx=fold_idx
        )

        val_results.append(best_val_dice)

        # Inference
        model.load_state_dict(best_model_state)
        t_dice, t_surf = test_inference(model, test_subjects, device)

        test_dice_scores.append(t_dice)
        test_surface_scores.append(t_surf)

        # LOGGING: Fold Summary
        logging.info(
            f"Fold {fold_id} Completed -> Val Dice: {best_val_dice:.4f} | Test Dice: {t_dice:.4f} | Test Surf: {t_surf:.4f}")

    avg_test_dice = np.mean(test_dice_scores)
    avg_test_surf = np.mean(test_surface_scores)

    # LOGGING: Final Summary
    logging.info("=== Cross-Validation Complete ===")
    logging.info(f"Average Test Dice: {avg_test_dice:.4f}")
    logging.info(f"Average Test Surface Dice: {avg_test_surf:.4f}")

    return val_results, avg_test_dice, avg_test_surf
