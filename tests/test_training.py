import pytest


def lr(trainer):
    return trainer.optimizer.param_groups[0]["lr"]


@pytest.mark.parametrize("schedule", ["Warmup", "WarmupPoly"])
def test_warmup_starts_below_the_initial_learning_rate(train, schedule):
    trainer = train(
        1, f"scheduler.name={schedule}", "scheduler.warmup_epochs=2", "scheduler.start_epoch=3"
    )  # stops halfway through the warmup

    assert lr(trainer) < 1e-2


def test_plateau_schedule_reduces_the_learning_rate(train):
    trainer = train(2, "scheduler.name=ReduceLROnPlateau", "+scheduler.factor=0.5", "+scheduler.patience=1")

    assert lr(trainer) == pytest.approx(0.5e-2)


def test_training_without_scheduler_saves_a_checkpoint(train, tmp_path):
    train(1, "scheduler.name=null")

    assert (tmp_path / "last_checkpoint.pytorch").is_file()


def test_resumed_training_continues_the_warmup(train, tmp_path):
    warmup = ["scheduler.name=WarmupPoly", "scheduler.warmup_epochs=3", "scheduler.start_epoch=4"]
    (tmp_path / "other").mkdir()
    uninterrupted = lr(train(2, *warmup, train_dir=tmp_path / "other"))

    train(1, *warmup)
    resumed = lr(train(2, *warmup))

    assert resumed == pytest.approx(uninterrupted)


def test_training_on_cpu_does_not_ask_for_cuda(train, recwarn):
    train(1, "mixed_precision=true")

    assert [str(w.message) for w in recwarn if "CUDA" in str(w.message)] == []


@pytest.mark.parametrize("schedule", ["WarmupPoly", "ReduceLROnPlateau"])
def test_checkpoints_load_without_unpickling_objects(train, tmp_path, schedule):
    import torch

    overrides = [f"scheduler.name={schedule}"] + (
        ["+scheduler.factor=0.5", "+scheduler.patience=1"] if "Plateau" in schedule else []
    )
    train(1, *overrides)

    state = torch.load(tmp_path / "last_checkpoint.pytorch", weights_only=True)
    resumed = train(2, *overrides)

    assert state["epoch"] == 1 and resumed.num_epoch == 2


def test_validation_keeps_the_best_checkpoint_across_resumes(train, tmp_path):
    import torch

    first = train(1, validation=True)
    assert (tmp_path / "best_checkpoint.pytorch").is_file()
    assert first.best_valid_loss == torch.load(tmp_path / "last_checkpoint.pytorch")["best_valid_loss"] < float("inf")

    resumed = train(2, validation=True)

    assert resumed.best_valid_loss <= first.best_valid_loss


def test_training_without_validation_keeps_no_best_checkpoint(train, tmp_path):
    train(1)

    assert not (tmp_path / "best_checkpoint.pytorch").exists()
