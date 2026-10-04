import os
import sys
import torch

from deepanepose.utils import get_optimizer_scheduler, get_model, load_training_config
from deepanepose.data.io import add_points_to_patient_data, readSplit, read_patient_data_base
from deepanepose.training.losses import get_loss_criterion

from deepanepose.training.trainer import create_trainer
from deepanepose.training.dataset import getDataloaders


def main():
    if len(sys.argv) != 2:
        sys.exit(f"Usage: {sys.argv[0]} path/to/Train001")
    config = load_training_config(sys.argv[1])

    workers = min(12, os.cpu_count() or 1)  # data loading processes, at most one per CPU
    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Model
    model = get_model(config, device=device)

    # Data Preparation
    ## Load patients
    train_list, valid_list, _ = readSplit(config["split_file"], config.get("base_dir"))
    ## Load training and validation patches
    train_db = add_points_to_patient_data(
        read_patient_data_base(
            pat_list=train_list,
            volume=config["volume"],
            pts_aneurysm=config["truth file"],
            normalize=config["normalize"],
            label="Training",
            workers=workers,
        ),
        config["negative patch centers"],
        nb=None,
    )
    valid_db = (
        add_points_to_patient_data(
            read_patient_data_base(
                pat_list=valid_list,
                volume=config["volume"],
                pts_aneurysm=config["truth file"],
                normalize=config["normalize"],
                label="Validation",
                workers=workers,
            ),
            config["negative patch centers"],
            nb=None,
        )
        if len(valid_list) > 0
        else None
    )
    # Create datasets and data loaders
    loaders, iterations = getDataloaders(train_db=train_db, valid_db=valid_db, config=config, workers=workers)
    # Create optimizer & scheduler
    warmup_steps = iterations["train"] * config["warmEpochs"] if "warmEpochs" in config else 0
    optimizer, scheduler = get_optimizer_scheduler(model=model, config=config, warmup_steps=warmup_steps)
    # Create Trainer and start/resume training
    trainer = create_trainer(
        config, device, model, optimizer, scheduler, get_loss_criterion(config["loss"]), loaders, iterations
    )
    trainer.fit()


if __name__ == "__main__":
    main()
