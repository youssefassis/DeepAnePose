import json

import pytest


@pytest.fixture
def evaluation_fold(tmp_path):
    """
    A trained fold ready for evaluation: P0001 and P0002 each have a 4mm aneurysm (aneurysms.csv, neck at x=-2) and two
    predictions (one on the aneurysm, one false positive); P0003 has no aneurysm and one false positive.
    Returns the predictions directory.
    """
    predictions_dir = tmp_path / "Work" / "Fold1" / "Predictions" / "10_epochs"
    predictions_dir.mkdir(parents=True)
    patients = []
    for name, has_aneurysm in (("P0001", True), ("P0002", True), ("P0003", False)):
        patient = tmp_path / "Data" / name
        patient.mkdir(parents=True)
        detections = {"0": {"center": [50.0, 0, 0], "radius": 2.0, "P1": [48.0, 0, 0], "P2": [52.0, 0, 0], "confidence": 0.6}}
        if has_aneurysm:
            (patient / "aneurysms.csv").write_text("x,y,z\n-2,0,0\n2,0,0\n")
            detections["1"] = {"center": [0.5, 0, 0], "radius": 2.0, "P1": [-1.5, 0, 0], "P2": [2.5, 0, 0], "confidence": 0.9}
        (predictions_dir / f"{name}.json").write_text(json.dumps(detections))
        patients.append(str(patient))

    split = {"training list": [], "validation list": patients[:1], "testing list": patients[1:]}
    (tmp_path / "fold1.json").write_text(json.dumps(split))
    config = {"split_file": str(tmp_path / "fold1.json"), "truth file": "aneurysms.csv"}
    (tmp_path / "Work" / "Fold1" / "ndl_config.json").write_text(json.dumps(config))
    return predictions_dir


@pytest.fixture
def model_config(tmp_path):
    """Config of a tiny model trained on CPU on 16-voxel patches of 8mm, without augmentation, with its checkpoints in tmp_path."""
    return {
        "patch_shape": [16, 16, 16], "patch_size": [0.5, 0.5, 0.5], "scales": [2], "batch_size": 2, "balancedbatch": True,
        "drop_last": False, "positive duplicates": 2, "nb neg patches per patient": None, "percentage of negative patches": 50,
        "flip probability": 0.5, "flip axes": [0],
        **{f"{kind} sample {aug}": None for kind in ("positive", "negative") for aug in ("shift", "rotation", "distortion", "scaling")},
        "model": "YOLO_Orientation", "num parameters": 6, "in_channels": 1, "n_base_filters": 4, "layer_order": "cbl", "depth": 4,
        "basic_module": "ExtResNetBlock", "pool_type": "conv", "loss": "YOLO_Regression_Cosine_Loss",
        "optimizer": "SGD", "momentum": 0.9, "weight_decay": 0, "lr": "WarmupPoly", "warmEpochs": 1,
        "initial_learning_rate": 1e-2, "min_lr": 1e-4, "start_epoch": 1, "stop_epoch": 4, "save model each n epoch": 10,
        "model_file": str(tmp_path / "last_checkpoint.pytorch"),
    }


@pytest.fixture
def train(model_config):
    """Trains (or resumes training) the tiny model up to n_epochs on two 40mm patients and returns its trainer."""
    import numpy as np

    from deepanepose.training.dataset import getDataloaders
    from deepanepose.training.losses import get_loss_criterion
    from deepanepose.training.trainer import create_trainer
    from deepanepose.utils import get_model, get_optimizer_scheduler

    rng = np.random.default_rng(0)
    patients = [
        {"dir": f"P000{i}", "data": rng.random((40, 40, 40)), "affine": np.eye(4),
         "aneurysms": np.array([[20.0, 20.0, 20.0], [21.0, 20.0, 20.0]]), "points": np.array([[8.0, 8.0, 8.0], [30.0, 30.0, 30.0]])}
        for i in (1, 2)
    ]

    def run(n_epochs, **overrides):
        config = {**model_config, "n_epochs": n_epochs, **overrides}
        loaders, iterations = getDataloaders(patients, None, config, workers=0)
        model = get_model(config)
        optimizer, scheduler = get_optimizer_scheduler(model, config, warmup_steps=iterations["train"] * config["warmEpochs"])
        trainer = create_trainer(config, "cpu", model, optimizer, scheduler, get_loss_criterion(config["loss"]), loaders, iterations)
        trainer.fit()
        return trainer

    return run
