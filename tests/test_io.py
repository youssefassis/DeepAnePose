import os

import nibabel as ni
import numpy as np

from deepanepose.data import io as dio


def test_linear_normalization_rescales_to_unit_range(tmp_path):
    vol = np.arange(27, dtype=np.float32).reshape((3, 3, 3)) + 10
    ni.save(ni.Nifti1Image(vol, np.eye(4)), tmp_path / "volume.nii.gz")

    data = dio.read_patient_data(str(tmp_path / "volume.nii.gz"), None, normalize="Linear")["data"]

    assert data.min() == 0 and data.max() == 1


def test_extract_points_script_handles_patients_with_and_without_aneurysm(tmp_path, monkeypatch):
    import json

    import pandas as pd

    from deepanepose.cli import extract_points

    monkeypatch.chdir(tmp_path)
    vol = np.random.default_rng(0).random((30, 30, 30)).astype(np.float32)
    for name, has_aneurysm in (("P0001", True), ("P0002", False)):
        patient = tmp_path / "Data" / name
        patient.mkdir(parents=True)
        ni.save(
            ni.Nifti1Image(vol, np.diag([4.0, 4.0, 4.0, 1.0])), patient / "noskull.nii.gz"
        )  # 120mm, room for 20mm spacing
        config = {"noskull volume": "noskull.nii.gz"}
        if has_aneurysm:
            (patient / "aneurysm.csv").write_text("x,y,z\n40,40,40\n44,40,40\n")
            config["pts aneurysm"] = "aneurysm.csv"
        (patient / "config.json").write_text(json.dumps(config))
    monkeypatch.setattr(
        "sys.argv", ["extract_points.py", str(tmp_path / "Data" / "P0001"), str(tmp_path / "Data" / "P0002")]
    )

    extract_points.main()

    for name in ("P0001", "P0002"):
        assert set(pd.read_csv(tmp_path / "Data" / name / "points.csv")["type"]) == {"Vessel", "Parenchyma"}
    assert os.getcwd() == str(tmp_path)  # the working directory is left as it was


def test_read_points_reports_a_missing_file(tmp_path):
    import pytest

    with pytest.raises(FileNotFoundError):
        dio.read_points_from_csv(tmp_path / "points.csv")


def test_relative_split_entries_are_taken_from_the_data_folder(tmp_path):
    import json

    split = tmp_path / "fold1.json"
    split.write_text(
        json.dumps({"training list": ["sub-001/ses-1"], "validation list": [], "testing list": ["/abs/P0001"]})
    )

    assert dio.read_split(str(split), "/data") == (["/data/sub-001/ses-1"], [], ["/abs/P0001"])
