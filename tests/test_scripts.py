import importlib

import pytest


@pytest.mark.parametrize("name", ["train", "predict"])
def test_script_without_training_directory_shows_its_usage(name, monkeypatch):
    monkeypatch.setattr("sys.argv", [f"{name}.py"])

    with pytest.raises(SystemExit, match="Usage"):
        importlib.import_module(name).main()
