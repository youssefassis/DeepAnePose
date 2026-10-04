import nibabel as ni
import numpy as np

import Data.IO as dio


def test_linear_normalization_rescales_to_unit_range(tmp_path):
    vol = np.arange(27, dtype=np.float32).reshape((3, 3, 3)) + 10
    ni.save(ni.Nifti1Image(vol, np.eye(4)), tmp_path / "volume.nii.gz")

    data = dio.read_patient_data(str(tmp_path / "volume.nii.gz"), None, normalize="Linear")["data"]

    assert data.min() == 0 and data.max() == 1


def test_extract_points_script_handles_patients_with_and_without_aneurysm(tmp_path, monkeypatch):
    import json

    import pandas as pd

    import extractPoints

    monkeypatch.chdir(tmp_path)  # extractPointsFromPatient changes the working directory
    vol = np.random.default_rng(0).random((30, 30, 30)).astype(np.float32)
    for name, has_aneurysm in (("P0001", True), ("P0002", False)):
        patient = tmp_path / "Data" / name
        patient.mkdir(parents=True)
        ni.save(ni.Nifti1Image(vol, np.diag([4.0, 4.0, 4.0, 1.0])), patient / "noskull.nii.gz")  # 120mm, room for 20mm spacing
        config = {"noskull volume": "noskull.nii.gz"}
        if has_aneurysm:
            (patient / "aneurysm.csv").write_text("x,y,z\n40,40,40\n44,40,40\n")
            config["pts aneurysm"] = "aneurysm.csv"
        (patient / "config.json").write_text(json.dumps(config))
    monkeypatch.setattr("sys.argv", ["extractPoints.py", str(tmp_path / "Data")])

    extractPoints.main()

    for name in ("P0001", "P0002"):
        assert set(pd.read_csv(tmp_path / "Data" / name / "points.csv")["type"]) == {"Vessel", "Parenchyma"}


def test_read_points_reports_a_missing_file(tmp_path):
    import pytest

    with pytest.raises(FileNotFoundError):
        dio.read_points_from_csv(tmp_path / "points.csv")
