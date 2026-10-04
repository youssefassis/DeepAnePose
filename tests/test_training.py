import pytest


def lr(trainer):
    return trainer.optimizer.param_groups[0]["lr"]


@pytest.mark.parametrize("schedule", ["Warmup", "WarmupPoly"])
def test_warmup_starts_below_the_initial_learning_rate(train, schedule):
    trainer = train(n_epochs=1, lr=schedule, warmEpochs=2, start_epoch=3)  # stops halfway through the warmup

    assert lr(trainer) < 1e-2


def test_plateau_schedule_reduces_the_learning_rate(train):
    trainer = train(n_epochs=2, lr="ReduceLROnPlateau", factor=0.5, patience=1)

    assert lr(trainer) == pytest.approx(0.5e-2)


def test_training_without_scheduler_saves_a_checkpoint(train, tmp_path):
    train(n_epochs=1, lr=None)

    assert (tmp_path / "last_checkpoint.pytorch").is_file()


def test_resumed_training_continues_the_warmup(train, tmp_path):
    warmup = {"lr": "WarmupPoly", "warmEpochs": 3, "start_epoch": 4}
    uninterrupted = lr(train(n_epochs=2, **warmup, model_file=str(tmp_path / "other" / "last_checkpoint.pytorch")))

    train(n_epochs=1, **warmup)
    resumed = lr(train(n_epochs=2, **warmup))

    assert resumed == pytest.approx(uninterrupted)
