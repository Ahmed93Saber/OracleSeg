import pandas as pd
import torch
from sklearn.model_selection import train_test_split, StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix
import os
import random
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple
from sklearn.metrics import auc, roc_curve
import itertools
import logging
import torchio as tio

from torchmetrics.functional import accuracy


def load_and_preprocess_data(geo_csv_path, curated_csv_path, label_col):
    """
    Loads GEO and curated datasets, merges them on patient identifiers, filters out rows with missing labels,
    and computes the number of weeks from a reference date.
    Args:
        geo_csv_path (str): Path to the GEO dataset CSV file.
        curated_csv_path (str): Path to the curated dataset CSV file.
        label_col (str): The column name for the labels in the curated dataset.
    Returns:
        pd.DataFrame: Preprocessed GEO dataset with labels and computed weeks.
        list: List of columns excluded from features.
    """
    geo_df = pd.read_csv(geo_csv_path)
    curated_df = pd.read_csv(curated_csv_path)
    geo_df = geo_df.merge(curated_df[['Patient ID', 'id', label_col]], on=['Patient ID', 'id'], how='left')
    geo_df = geo_df[geo_df[label_col].notna()]
    geo_df = add_num_weeks_column(geo_df, 'CROSSING_TIME_POINT')
    return geo_df


def load_radiomics_splits():
    # TODO: Delete hardcoded paths
    train_df = pd.read_csv('./dataframes/train_radiomics.csv')
    test_df = pd.read_csv('./dataframes/test_radiomics.csv')

    return train_df, test_df

def split_and_scale_data(df, label_col, test_size=0.2,
                         random_state=0, is_radiomics=False, external_test_df=None):
    """
    Splits the dataframe into train and test sets and scales feature columns.
    """
    if is_radiomics:
        categorical_vols = ['1+2.0', '2+2.0', '2+3.0', '2+1.0']
        train_df, test_df = load_radiomics_splits()

    else:
        categorical_vols = []
        train_df, test_df = train_test_split(df, test_size=test_size,
                                             random_state=random_state, stratify=df[label_col])
    # scaler = StandardScaler()
    # train_df[feature_cols] = scaler.fit_transform(train_df[feature_cols])
    # test_df[feature_cols] = scaler.transform(test_df[feature_cols])
    #
    # # Add categorical columns to the splits
    # for col in categorical_vols:
    #     train_df[col] = train_df[col]
    #     test_df[col] = test_df[col]


    return train_df, test_df


def split_and_scale_data_type(df, label_col, random_state=42):
    # 0. Define Features (X), Target (y), and Groups
    y = df[label_col]
    X = df.drop(label_col, axis=1)
    groups = df['Patient ID']

    # 1. Setup (assuming 'df', 'X', 'y', and 'groups' are defined as before)
    cv_splitter = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=random_state)

    # 2. Extract ONE fold to be the Hold-out Test Set
    # next() grabs the first iteration of the generator
    train_idxs, test_idxs = next(cv_splitter.split(X, y, groups=groups))

    X_train_full = X.iloc[train_idxs]
    y_train_full = y.iloc[train_idxs]
    groups_train = groups.iloc[train_idxs]  # Important: Slice groups too!

    X_test = X.iloc[test_idxs]
    y_test = y.iloc[test_idxs]

    # 3. combine X and y back into DataFrames for train and test
    train_df = X_train_full.copy()
    train_df[label_col] = y_train_full

    test_df = X_test.copy()
    test_df[label_col] = y_test

    return train_df, test_df, groups_train

def add_num_weeks_column(df, date_col, reference_date=None):
    """
    Adds a 'num_weeks' column to the dataframe, representing the number of weeks from a reference date.
    """
    if reference_date is None:
        reference_date = pd.to_datetime('1900-01-01')
    else:
        reference_date = pd.to_datetime(reference_date)
    df[date_col] = pd.to_datetime(df[date_col].astype(str), format='%Y-%m-%d')
    df["num_weeks"] = ((df[date_col] - reference_date).dt.days // 7).astype(int)
    return df


def calculate_sensitivity_specificity(y_true, y_pred):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    sensitivity = tp / (tp + fn)
    specificity = tn / (tn + fp)
    return sensitivity, specificity


def log_optuna_metrics(trial, metrics, is_test=False, is_external=False):
    """
    Logs the metrics to Optuna.
    """
    for metric, value in metrics.items():
        if is_test and not is_external:
            trial.set_user_attr(f"test_{metric}", value)
        elif is_test and is_external:
            trial.set_user_attr(f"external_test_{metric}", value)
        else:
            trial.set_user_attr(metric, value)


def save_models(models_stat_dicts, trial, mean_metrics, saving_path="weights"):


    if len(models_stat_dicts) == 5 and mean_metrics['accuracy'] > 0.7 and mean_metrics['f1'] > 0.7:

        for fold, state_dict in enumerate(models_stat_dicts):
            torch.save(state_dict, f"{saving_path}/{trial.user_attrs['modality']}/model_fold_{fold}_trial_{trial.number}.pth")


def set_random(seed=1, deterministic=True):
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = deterministic
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)


def plot_auc(predictions, ground_truths, num, mode, save=False):
    """
    Plot the ROC curve and display in the upper third of an A4 paper.

    :param num: trial number
    :param predictions: list of the list the predictions for each fold
    :param ground_truths: list of the list of the ground truth for each fold
    :param mode: mode of the model
    :param save: save the plot
    :return: plot the ROC curve
    """
    # Set font sizes
    axis_label_fontsize = 32
    legend_fontsize = 32
    tick_label_fontsize = 32
    line_width = 4

    # Prepare figure with A4 upper third dimensions
    fig = plt.figure(figsize=(15, 15))  # A4 width, one-third A4 height

    colors = itertools.cycle(['blue', 'darkorange', 'purple', 'green', 'black', 'red'])
    fprs = []
    tprs = []
    aucs = []

    # Compute ROC curves for each fold and store them
    for fold_pred, fold_gt in zip(predictions, ground_truths):
        fpr, tpr, _ = roc_curve(fold_gt, fold_pred)
        auc_value = auc(fpr, tpr)
        fprs.append(fpr)
        tprs.append(tpr)
        aucs.append(auc_value)

    # Define a common set of points for averaging across all folds
    mean_fpr = np.linspace(0, 1, 100)
    interp_tprs = []

    # Plot individual ROC curves in faded lines
    for i, (fpr, tpr) in enumerate(zip(fprs, tprs)):
        plt.plot(fpr, tpr, color=next(colors), lw=line_width, alpha=0.2,  # Faded line for individual folds
                 label=f'ROC curve fold {i + 1} (AUC = {aucs[i]:0.2f})')
        interp_tpr = np.interp(mean_fpr, fpr, tpr)
        interp_tpr[0] = 0.0
        interp_tprs.append(interp_tpr)

    # Calculate mean TPR and standard deviation
    mean_tpr = np.mean(interp_tprs, axis=0)
    mean_tpr[-1] = 1.0
    mean_auc = auc(mean_fpr, mean_tpr)

    # Plot mean ROC curve in bold solid line
    plt.plot(mean_fpr, mean_tpr, color='black', lw=line_width + 2, linestyle='-',
             label=f'Mean ROC (AUC = {str(mean_auc)[:4]})')

    # Plot diagonal line for reference
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    # Labels, title, and grid
    plt.xlabel('False Positive Rate', fontsize=axis_label_fontsize)
    plt.ylabel('True Positive Rate', fontsize=axis_label_fontsize)
    plt.tick_params(axis='both', which='major', labelsize=tick_label_fontsize)
    plt.legend(loc='lower right', fontsize=legend_fontsize)
    plt.grid()

    # Adjust layout to fit in upper third of A4
    plt.subplots_adjust(top=0.85, bottom=0.15)  # Adjust padding to fit text

    # Save or show plot
    if save:
        os.makedirs('./plots/production/', exist_ok=True)
        plt.savefig(f'./plots/production/roc_{num}_{mode}.svg', bbox_inches='tight')
    plt.show()

def load_pretrained_model(pretrained_model_path, pre_trained_model):
    state_dict = torch.load(pretrained_model_path, map_location="cpu", weights_only=False)
    if any(k.startswith("module.") for k in state_dict.keys()):
        state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
    pre_trained_model.load_state_dict(state_dict, strict=False)

    return pre_trained_model


def enforce_features(source_df: pd.DataFrame, target_dfs: list, features_cols: list, is_scalable=False):
    """
    Enforce features from source_df to target_dfs. This is done by merging the source_df with each target_df
    Args:
        source_df: dataframe containing the features to be enforced
        target_dfs: [train_df, test_df]
        features_cols: feature columns to be enforced
        is_scalable: if True, scale the features using StandardScaler

    """
    on_columns = ['Patient ID', 'id']
    for i, target_df in enumerate(target_dfs):
        target_df = target_df.merge(source_df[on_columns + features_cols], on=on_columns, how='left')
        target_dfs[i] = target_df


    if not is_scalable:
        return target_dfs
    scaler = StandardScaler()
    target_dfs[0][features_cols] = scaler.fit_transform(target_dfs[0][features_cols])
    target_dfs[1][features_cols] = scaler.transform(target_dfs[1][features_cols])

    return target_dfs


def log_metrics_stats(trial, mean_metrics_dict, std_test_metrics, is_external=False):

    prefix = "External " if is_external else ""

    logging.info(
        f"Mean Test Metrics Across All Folds for trial {trial.number}: "
        f"{prefix}Loss: {mean_metrics_dict['loss']:.4f} "
        f"{prefix}Acc: {mean_metrics_dict['accuracy']:.4f}±{std_test_metrics['accuracy']:.4f}, "
        f"{prefix}F1: {mean_metrics_dict['f1']:.4f}±{std_test_metrics['f1']:.4f}, "
    )


def find_optimal_threshold(y_true, y_pred):
    """Calculate optimal threshold using Youden's Index"""
    thresholds = np.arange(0.1, 0.9, 0.01)
    best_threshold = 0.5
    best_youden = 0

    for threshold in thresholds:
        pred_labels = (y_pred > threshold).astype(int)
        sensitivity, specificity = calculate_sensitivity_specificity(y_true, pred_labels)
        youden = sensitivity + specificity - 1

        if youden > best_youden:
            best_youden = youden
            best_threshold = threshold

    return best_threshold


def masking_function(x):
    """Return a boolean mask where values > 0 are True."""
    return x > 0


def load_and_preprocess_image(image_path, img_size=(1, 64, 64, 64)):
    transform = tio.Compose([
        tio.ToCanonical(),
        tio.Resample(1),
    #     # Clamp intensity
    #     tio.RescaleIntensity(
    #     out_min_max=(0, 1),
    #     percentiles=(1, 99),
    #     masking_method=masking_function       # to compute percentiles within brain mask
    # ),
    #     tio.ZNormalization(masking_method=masking_function),
    ])
    image = transform(image_path)
    # print(image.data.shape)
    if image.shape != img_size:
        # pad to the required size
        image = tio.CropOrPad(img_size[-1])(image)
    # image.plot()
    return image.data.unsqueeze(0)  # Add batch dimension


def initialize_pretrained_model(model_instance, checkpoint_path, unfreeze_last_n=0):
    """
    Initialize a model from a checkpoint and freeze/unfreeze specific ViT layers.
    Args:
        model_instance: The instance of the model class (e.g., UNETR)
        checkpoint_path: Path to the checkpoint file.
        unfreeze_last_n: Number of the last ViT blocks to keep trainable.
                         0 means the whole ViT backbone is frozen.
    Returns:
        The model loaded with weights and correct requires_grad states.
    """
    model = model_instance
    if os.path.isfile(checkpoint_path):
        vit_dict = torch.load(checkpoint_path, weights_only=False)
        vit_weights = vit_dict['model_state_dict']

        # Filter out weights not in the ViT backbone
        model_dict = model.vit.state_dict()
        vit_weights = {k: v for k, v in vit_weights.items() if k in model_dict}
        model_dict.update(vit_weights)
        model.vit.load_state_dict(model_dict)
        del model_dict, vit_weights, vit_dict
        print("Pretrained Weights Successfully Loaded!")

        # --- FREEZING LOGIC ---

        # 1. Freeze everything in the ViT backbone by default (including patch_embedding)
        for param in model.vit.parameters():
            param.requires_grad = False

        # 2. Unfreeze the last N blocks (if N > 0)
        if unfreeze_last_n > 0:
            for block in model.vit.blocks[-unfreeze_last_n:]:
                for param in block.parameters():
                    param.requires_grad = True

            # Unfreeze the final norm layer
            for param in model.vit.norm.parameters():
                param.requires_grad = True

        elif unfreeze_last_n == 0:
            print("=> ViT backbone is completely frozen.")

        return model
    else:
        raise FileNotFoundError(f"No checkpoint found at {checkpoint_path}")
