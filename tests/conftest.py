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
