import json
import subprocess
import sys
from pathlib import Path

import pytest
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

CONFIG_DIR = str(Path(__file__).parents[1] / "deepanepose" / "configs")

# a tiny model trained on CPU on 16-voxel patches of 8 mm, without augmentation
TINY_TRAINING = [
    "name=exp",
    "data.split_file=split.json",
    "data.patch_shape=[16,16,16]",
    "data.patch_size=[8,8,8]",
    "model.f_maps=4",
    "batch_size=2",
    "drop_last=false",
    "negative_patches=50",
    "num_workers=0",
    "validation_batch_size=2",
    "augmentation.positive.duplicates=2",
    *[
        f"augmentation.{kind}.{aug}=null"
        for kind in ("positive", "negative")
        for aug in ("shift", "rotation", "distortion")
    ],
    "optimizer.momentum=0.9",
    "optimizer.weight_decay=0",
    "scheduler.warmup_epochs=1",
    "scheduler.start_epoch=1",
    "scheduler.stop_epoch=4",
]


def compose_config(name, overrides=()):
    with initialize_config_dir(config_dir=CONFIG_DIR, version_base="1.3"):
        return compose(config_name=name, overrides=list(overrides))


def save_training_config(train_dir, cfg, launch_dir):
    """Writes the .hydra folder of a training launched from launch_dir, as deepanepose-train does"""
    (train_dir / ".hydra").mkdir(parents=True, exist_ok=True)
    OmegaConf.save(cfg, train_dir / ".hydra" / "config.yaml")
    OmegaConf.save({"hydra": {"runtime": {"cwd": str(launch_dir)}}}, train_dir / ".hydra" / "hydra.yaml")


def run_command(module, *overrides, **kwargs):
    """Runs a deepanepose.cli module as its installed command does"""
    return subprocess.run(
        [sys.executable, "-m", f"deepanepose.cli.{module}", *map(str, overrides)],
        capture_output=True,
        text=True,
        **kwargs,
    )


@pytest.fixture
def evaluation_fold(tmp_path):
    """
    A trained fold ready for evaluation: P0001 and P0002 each have a 4mm aneurysm (aneurysms.csv, neck at x=-2) and two
    predictions (one on the aneurysm, one false positive); P0003 has no aneurysm and one false positive.
    Returns the training and predictions directories.
    """
    train_dir = tmp_path / "Data" / "0Work" / "fold1"
    predictions_dir = train_dir / "prediction" / "10_epochs"
    predictions_dir.mkdir(parents=True)
    patients = []
    for name, has_aneurysm in (("P0001", True), ("P0002", True), ("P0003", False)):
        patient = tmp_path / "Data" / name
        patient.mkdir(parents=True)
        detections = {
            "0": {"center": [50.0, 0, 0], "radius": 2.0, "P1": [48.0, 0, 0], "P2": [52.0, 0, 0], "confidence": 0.6}
        }
        if has_aneurysm:
            (patient / "aneurysms.csv").write_text("x,y,z\n-2,0,0\n2,0,0\n")
            detections["1"] = {
                "center": [0.5, 0, 0],
                "radius": 2.0,
                "P1": [-1.5, 0, 0],
                "P2": [2.5, 0, 0],
                "confidence": 0.9,
            }
        (predictions_dir / f"{name}.json").write_text(json.dumps(detections))
        patients.append(name)

    split = {"training list": [], "validation list": patients[:1], "testing list": patients[1:]}
    (tmp_path / "fold1.json").write_text(json.dumps(split))
    cfg = compose_config("train", ["data_dir=Data", "data.split_file=fold1.json", "name=fold1"])
    save_training_config(train_dir, cfg, launch_dir=tmp_path)
    return train_dir, predictions_dir


@pytest.fixture
def train(tmp_path):
    """
    Trains (or resumes training) the tiny model up to n_epochs on two 40mm patients, with Hydra overrides of the
    training config, in train_dir (tmp_path by default), validating on the same patients with validation=True;
    returns its trainer.
    """
    import numpy as np
    from hydra.utils import instantiate

    from deepanepose.training.dataset import get_dataloaders
    from deepanepose.training.losses import get_loss_criterion
    from deepanepose.training.trainer import create_trainer
    from deepanepose.utils import create_lr_scheduler, get_model

    rng = np.random.default_rng(0)
    patients = [
        {
            "dir": f"P000{i}",
            "data": rng.random((40, 40, 40)),
            "affine": np.eye(4),
            "aneurysms": np.array([[20.0, 20.0, 20.0], [21.0, 20.0, 20.0]]),
            "points": np.array([[8.0, 8.0, 8.0], [30.0, 30.0, 30.0]]),
        }
        for i in (1, 2)
    ]

    def run(n_epochs, *overrides, train_dir=tmp_path, validation=False):
        cfg = compose_config("train", [*TINY_TRAINING, f"data_dir={tmp_path}", f"epochs={n_epochs}", *overrides])
        loaders, iterations = get_dataloaders(patients, patients if validation else None, cfg)
        model = get_model(cfg.model)
        optimizer = instantiate(cfg.optimizer, params=model.parameters(), _convert_="all")
        warmup_epochs = cfg.scheduler.warmup_epochs
        scheduler = create_lr_scheduler(optimizer, cfg.scheduler, cfg.epochs, iterations["train"] * warmup_epochs)
        trainer = create_trainer(
            str(train_dir),
            "cpu",
            model,
            optimizer,
            scheduler,
            get_loss_criterion(cfg.loss),
            loaders,
            iterations,
            max_num_epochs=cfg.epochs,
            save_every=cfg.save_every,
            warmup_epochs=warmup_epochs,
            mixed_precision=cfg.mixed_precision,
        )
        trainer.fit()
        return trainer

    return run
