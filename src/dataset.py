import torch
import pandas as pd
import numpy as np
from Tools.scripts.objgraph import ignore
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import StratifiedKFold, StratifiedGroupKFold
from typing import List, Tuple, Optional
from glob import glob
import torchio as tio
import os



class BasePatientDataset(Dataset):
    """
    Shared utilities for clinical and imaging datasets.
    """
    def __init__(self, dataframe: pd.DataFrame, label_col: str = 'label-1RN-0Normal'):
        self.dataframe = dataframe
        self.label_col = label_col


    def get_label_tensor(self, row) -> torch.Tensor:
        return torch.tensor(row[self.label_col], dtype=torch.float32)

    @staticmethod
    def get_dict_key(row) -> str:
        patient_id = row["Patient ID"]
        met_id = row["id"]
        scan_date = row["scan_date"].split()[0]
        return f"{patient_id}_{scan_date}_{met_id}"


class ClinicalDataset(BasePatientDataset):
    def __init__(self, dataframe: pd.DataFrame, columns_to_drop: List[str],
                 is_external: bool = False, label_col: str = 'label-1RN-0Normal'):
        super().__init__(dataframe, label_col)
        self.columns_to_drop = columns_to_drop
        self.is_external = is_external

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        row = self.dataframe.iloc[index]
        features = row.drop(self.columns_to_drop, errors='ignore').values.astype('float32')
        features_tensor = torch.tensor(features, dtype=torch.float32)
        label_tensor = self.get_label_tensor(row)
        return features_tensor, label_tensor

    def __len__(self) -> int:
        return len(self.dataframe)


class ImagingDataset(BasePatientDataset):
    def __init__(self, dataframe: pd.DataFrame, data_dir: str,
                 is_external: bool = False, ext_data_dir: Optional[str] = None, label_col: str = 'label-1RN-0Normal'):

        super().__init__(dataframe, label_col)

        self.is_external = is_external

        if is_external and ext_data_dir is not None:
            self.img_seq = np.load(ext_data_dir, allow_pickle=True).item()
        elif not is_external:
            self.img_seq = np.load(data_dir, allow_pickle=True).item()



    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        row = self.dataframe.iloc[index]

        if self.is_external:
            dict_key = row["Patient ID"]
        else:
            dict_key = self.get_dict_key(row)

        try:
            img_tensor = self.img_seq[dict_key][...].clone().detach().float()

        except KeyError:
            print(f"KeyError: {dict_key} not found in img_seq.")
            img_tensor = torch.zeros((2,64,64,64))


        label_tensor = self.get_label_tensor(row)
        return img_tensor, label_tensor

    def __len__(self) -> int:
        return len(self.dataframe)


class MultimodalDataset(BasePatientDataset):
    def __init__(self, dataframe: pd.DataFrame, data_dir: str, columns_to_drop: List[str],
                 is_external: bool = False, ext_data_dir: Optional[str] = None, label_col: str = 'label-1RN-0Normal'):


        super().__init__(dataframe, label_col)
        self.columns_to_drop = columns_to_drop
        self.is_external = is_external

        if is_external and ext_data_dir is not None:
            self.img_seq = np.load(ext_data_dir, allow_pickle=True).item()
        elif not is_external:
            self.img_seq = np.load(data_dir, allow_pickle=True).item()

    def __getitem__(self, index: int) -> Tuple[Tuple[torch.Tensor, torch.Tensor], torch.Tensor]:
        row = self.dataframe.iloc[index]

        # Clinical features
        clinical_data = row.drop(self.columns_to_drop, errors='ignore').values.astype('float32')
        clinical_tensor = torch.tensor(clinical_data, dtype=torch.float32)

        if self.is_external:
            dict_key = row["Patient ID"]
        else:
            dict_key = self.get_dict_key(row)

        try:
            img_tensor = self.img_seq[dict_key][...].clone().detach().float()
        except KeyError:
            print(f"KeyError: {dict_key} not found in img_seq.")
            img_tensor = torch.zeros((2,64,64,64))

        img_tensor = img_tensor[0, ...].unsqueeze(0) # TODO

        label_tensor = self.get_label_tensor(row)
        return (img_tensor, clinical_tensor), label_tensor

    def __len__(self) -> int:
        try:
            return len(self.dataframe)
        except IndexError:
            print("IndexError: DataFrame is empty.")



def _k_fold_groups(df, label_column, groups_train, n_splits=5, random_state=42):
    y_train_full = df[label_column]
    X_train_full = df[~df[label_column].isna()]
    cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
    for fold, (t_idx, v_idx) in enumerate(cv.split(X_train_full, y_train_full, groups=groups_train)):
        train_data = df.iloc[t_idx]
        val_data = df.iloc[v_idx]


def create_dataloaders(
        train_df: pd.DataFrame,
        label_column: str,
        exclude_columns: list,
        batch_size: int,
        n_splits: int = 5,
        group: str = None,  # <--- NEW ARGUMENT
        dataset_cls=None,  # defaulting to None for syntax, assumed defined
        dataset_kwargs: dict = None,
        random_state: int = 42
):
    """
    Creates k-fold dataloaders.
    - If group_column is None: Uses standard StratifiedKFold (Random split).
    - If group_column is provided: Uses StratifiedGroupKFold (Patient-aware split).
    """
    if dataset_kwargs is None:
        dataset_kwargs = {}

    # 1. Determine which Feature Columns to keep (metadata)
    feature_columns = [col for col in train_df.columns if col not in exclude_columns]

    # 2. Select the Splitter and Grouping Data
    if group is not None:
        # New Functionality: Group-aware splitting
        print(f"Using StratifiedGroupKFold on group: Patient ID")
        cv = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
        groups = group
    else:
        # Old Functionality: Standard Stratified splitting
        print("Using standard StratifiedKFold")
        cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
        groups = None

    dataloaders = {}

    # 3. Generate Splits
    # Note: sklearn's .split() handles 'groups=None' gracefully for StratifiedKFold
    y = train_df[label_column]

    for fold, (train_idx, val_idx) in enumerate(cv.split(train_df, y, groups=groups)):
        # Use .iloc because split returns positional indices
        train_data = train_df.iloc[train_idx]
        val_data = train_df.iloc[val_idx]

        # 4. Instantiate Datasets and Loaders
        train_loader = DataLoader(
            dataset_cls(train_data, **dataset_kwargs),
            batch_size=batch_size,
            shuffle=True,
            pin_memory=True
        )
        val_loader = DataLoader(
            dataset_cls(val_data, **dataset_kwargs),
            batch_size=batch_size,
            shuffle=False,
            pin_memory=True
        )

        dataloaders[fold] = {'train': train_loader, 'val': val_loader}

    return dataloaders, feature_columns


def create_subjects_list(pairs_list: List[Tuple[str, str]]) -> List[tio.Subject]:
    """
    Convert a list of (t1c_path, seg_path) into TorchIO Subject objects.
    """
    subjects = []
    for t1c_path, seg_path in pairs_list:
        mask = tio.LabelMap(seg_path)
        mask.data = torch.clamp(mask.data, max=1.0)
        if mask.data.sum() == 0:
            print(f"Skipping subject with empty mask: {seg_path}")
            continue

        subject = tio.Subject(
            t1c=tio.ScalarImage(t1c_path),
            mask=mask
        )
        subjects.append(subject)
    return subjects


def create_tio_dataloaders_molab(molab_df_path, molab_data_dir, training_split_ratio=0.8,
                                patch_size=64, samples_per_volume=24, batch_size=8, random_state=42, met_ration=0.5):



    patch_size_3d = (patch_size, patch_size, patch_size)

    data_list = []
    df = pd.read_csv(molab_df_path)
    for idx, row in df.iterrows():
        patient_id = row["Patient ID"]
        patient_id_str = f"0{patient_id}N"
        date = pd.to_datetime(row["scan_date"])
        patient_scans_dir = os.path.join(molab_data_dir, patient_id_str)
        scan_path = glob(os.path.join(patient_scans_dir, f"0{patient_id}_{date.strftime('%Y%m%d')}*_img_bfc_final.nii.gz"))[0]
        seg_path = glob(os.path.join(patient_scans_dir, f"0{patient_id}_{date.strftime('%Y%m%d')}*_msk_registered.nii"))[0]

        # check if paths exists
        if not os.path.exists(scan_path) or not os.path.exists(seg_path):
            print(f"Scan or segmentation not found for Patient ID: {patient_id}, Date: {date}")
            continue

        data_list.append((scan_path, seg_path))

    subjects = create_subjects_list(data_list)
    num_subjects = len(data_list)
    num_training_subjects = int(training_split_ratio * num_subjects)
    num_validation_subjects = num_subjects - num_training_subjects

    num_split_subjects = num_training_subjects, num_validation_subjects
    training_subjects, test_subjects = torch.utils.data.random_split(subjects,
                                                                     num_split_subjects,
                                                                     generator=torch.Generator().manual_seed(random_state))


    # perform 5-fold cross-validation split on training subjects
    training_subjects = list(training_subjects)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state)

    # Create dummy labels for stratification (can use patient ID or another criterion)
    train_labels = np.ones(len(training_subjects))

    fold_dataloaders = {}

    for fold, (train_idx, val_idx) in enumerate(cv.split(training_subjects, train_labels)):
        fold_train_subjects = [training_subjects[i] for i in train_idx]
        fold_val_subjects = [training_subjects[i] for i in val_idx]

        pre_processing = tio.Compose([
            tio.Pad((32, 32, 32)),
            tio.ZNormalization(masking_method=masking_function),
        ])

        # Create datasets with transforms
        fold_train_dataset = tio.SubjectsDataset(fold_train_subjects, transform=pre_processing)
        fold_val_dataset = tio.SubjectsDataset(fold_val_subjects, transform=pre_processing)

        # Create patch-based queues
        fold_train_queue = tio.Queue(
            subjects_dataset=fold_train_dataset,
            max_length=8,
            num_workers=0,
            samples_per_volume=8,
            sampler=tio.data.LabelSampler(
                patch_size=patch_size_3d,
                label_name="mask",
                label_probabilities={0: 1 - float(met_ration), 1: float(met_ration)},
            ),
            shuffle_subjects=True,
            shuffle_patches=True,
        )

        fold_val_queue = tio.Queue(
            subjects_dataset=fold_val_dataset,
            max_length=8,
            num_workers=0,
            samples_per_volume=8,
            sampler=tio.data.LabelSampler(patch_size_3d),
            shuffle_subjects=False,
            shuffle_patches=False,
        )

        # Create dataloaders
        fold_train_loader = DataLoader(
            fold_train_queue,
            batch_size=batch_size,
            num_workers=0
            # shuffle=True is usually not needed here as the Queue handles shuffling,
            # but usually SubjectsLoader handles the collate_fn automatically.
        )

        fold_val_loader = DataLoader(
            fold_val_queue,
            batch_size=batch_size,
            num_workers=0
        )


        fold_dataloaders[fold] = {
            'train': fold_train_loader,
            'val': fold_val_loader
        }

    return fold_dataloaders, test_subjects




class EvalFSRT(Dataset):
    """
    Dataset for evaluation purposes
    """
    def __init__(self, df_path: str, sheet_name: str, with_seg=False):


        self.df_path = df_path
        self.df = pd.read_excel(df_path, sheet_name=sheet_name)
        mask = self.df['1_ok_0_not'] != 0
        self.df = self.df[mask].reset_index(drop=True)
        self.with_seg = with_seg

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = row['Abs_path']
        seg_path = img_path.replace('scan', 'segmentation')
        label = row['label-1RN-0Normal']

        img_id = os.path.basename(img_path).split('_cropped')[0]

        # Load and preprocess the image
        img = tio.ScalarImage(img_path)
        input_tensor = load_and_preprocess_image(img)

        seg = tio.ToCanonical()(tio.LabelMap(seg_path)).data
        seg = (seg > 0).to(seg.dtype)
        if seg.shape != input_tensor.shape:
            seg = tio.CropOrPad(input_tensor.shape[-1])(seg)

        x_in = None
        if self.with_seg:
            try:
                x_in = torch.cat((input_tensor, seg.unsqueeze(0)), dim=1)  # Stack along the channel dimension
            except Exception as e:
                print(f"Error with stacking tensors: {e}")

        else:
            x_in = input_tensor

        return x_in, img_id, label



def masking_function(x):
    """Return a boolean mask where values > 0 are True."""
    return x > 0


def load_and_preprocess_image(image_path, img_size=(1, 64, 64, 64)):
    transform = tio.Compose([
        tio.ToCanonical(),
        # tio.Resample(1),
        # Clamp intensity
        # tio.RescaleIntensity(
        #     out_min_max=(0, 1),
        #     percentiles=(1, 99),
        #     masking_method=masking_function  # to compute percentiles within brain mask
        # ),
        # tio.ZNormalization(masking_method=masking_function),
    ])
    image = transform(image_path)
    if image.shape != img_size:
        # pad to the required size
        image = tio.CropOrPad(img_size[-1])(image)
    # image.plot()
    return image.data.unsqueeze(0)  # Add batch dimension