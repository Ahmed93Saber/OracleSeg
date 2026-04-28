import gc
import logging
import yaml
import numpy as np
import torch
import optuna
from monai.networks.nets import UNETR

# Internal project imports
from src.dataset import create_tio_dataloaders_molab
from src.utils import initialize_pretrained_model, set_random
from src.Patch_processing_train import run_patch_segmentation_cv


class SegmentationObjective:
    def __init__(self, config, epochs=300, patience=15):
        self.config = config
        self.epochs = epochs
        self.patience = patience
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Pull paths from config once
        self.csv_path = "dataframes/primary_tumor_types_3_cls.csv"
        self.molab_data_dir = r"S:\AI_Radiation\#Datasets\BrainMetData\MOLAB"
        self.pretrained_path = config['data']["pretrained_model_path"]

    def __call__(self, trial):
        # 1. Hyperparameter Sampling
        params = {
            "batch_size": trial.suggest_categorical("batch_size", [8, 16]),
            "layers_to_unfreeze": trial.suggest_int("layers_to_unfreeze", 8, 11),
            "met_ratio": trial.suggest_categorical("met_ration", [0.4, 0.5, 0.6, 0.7]),
            "lr": trial.suggest_float("learning_rate", 1e-5, 1e-2, log=True),
            "rs": trial.suggest_int("random_state", 0, 10),
            "wd": trial.suggest_float("weight_decay", 1e-5, 1e-2, log=True),
        }


        # 2. Data Loading
        fold_dataloaders, test_subjects = create_tio_dataloaders_molab(
            self.csv_path,
            self.molab_data_dir,
            batch_size=params["batch_size"],
            random_state=params["rs"],
            met_ration=params["met_ratio"]
        )

        # 3. Model Factory
        def model_factory():
            model = UNETR(
                in_channels=1,
                out_channels=1,
                img_size=(64, 64, 64),
                feature_size=16,
                hidden_size=600,
                num_heads=12,
                res_block=True,
                mlp_dim=2048,
            )
            return initialize_pretrained_model(
                model,
                self.pretrained_path,
                unfreeze_last_n=params["layers_to_unfreeze"]
            )

        # 4. Execution
        try:
            fold_results, test_dice, test_sdice = run_patch_segmentation_cv(
                model_factory,
                fold_dataloaders,
                test_subjects,
                self.device,
                num_epochs=self.epochs,
                patience=self.patience,
                lr=params["lr"],
                wd=params["wd"]
            )

            # Metrics
            mean_val_dice = float(np.mean(fold_results))
            trial.set_user_attr("test_dice", float(test_dice))
            trial.set_user_attr("test_surface_dice", float(test_sdice))

            return mean_val_dice

        finally:
            # 5. Cleanup (Ensures memory is freed even if the trial crashes)
            del fold_dataloaders
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()


def main(num_epochs):
    # Setup
    logging.basicConfig(
        filename='training_logs_imaging_seg.log',
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    set_random()

    # Load Config
    with open("config.yml", "r") as f:
        config = yaml.safe_load(f)

    modality = "imaging"

    # Optuna Logic
    study = optuna.create_study(
        study_name="pretrained-seg",
        direction="maximize"
    )

    objective = SegmentationObjective(config, epochs=num_epochs)

    study.optimize(objective, n_trials=120)

    # Export Results
    df = study.trials_dataframe()
    df.to_csv(f'optuna_results/optuna_results_seg_{modality}_cv.csv', index=False)
    print("Optimization complete. Results saved.")


if __name__ == "__main__":
    num_epochs = 500
    main(num_epochs)