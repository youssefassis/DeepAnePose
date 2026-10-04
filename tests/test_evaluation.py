import matplotlib
import pandas as pd

matplotlib.use("Agg")

import evaluate  # noqa: E402


def test_evaluate_script_writes_the_detection_table_and_figure(evaluation_fold, monkeypatch):
    monkeypatch.setattr("sys.argv", ["evaluate.py", str(evaluation_fold) + "/"])

    evaluate.main()

    table = pd.read_csv(evaluation_fold / "detection_evaluation@0.1.csv")
    assert table["TP"].sum() == 2 and table["FP"].sum() == 3
    assert (evaluation_fold / "Evaluation@0.1.png").is_file()


def test_detections_are_matched_by_index_to_the_most_overlapping_aneurysm():
    import numpy as np

    from deepanepose.inference.evaluation import get_CM_dict

    def sphere(x, r):  # center, radius, P1, P2 along x
        return [x, 0, 0, r, x - r, 0, 0, x + r, 0, 0]

    truths = np.array([sphere(0, 2), sphere(3, 2), sphere(50, 2)])  # the first and last have the same size
    predictions = np.array([sphere(2.5, 2) + [0.9], sphere(50, 2) + [0.8], sphere(50.5, 2) + [0.7]])

    detections, fns = get_CM_dict(predictions, truths, iou_thr=0.1, confidence_thr=0.01, pat_name="P0001", verbose=False)

    assert [(d["TP"], d.get("GT Center", [None])[0]) for d in detections] == [(1, 3), (1, 50), (0, 50)]
    assert [fn["Center"][0] for fn in fns] == [0]


def test_evaluate_script_without_false_or_true_positives(evaluation_fold, monkeypatch):
    import json

    for kept in ("true positives", "nothing"):
        for prediction_file in evaluation_fold.glob("*.json"):
            detections = json.loads(prediction_file.read_text())
            keep = {k: d for k, d in detections.items() if d["confidence"] == 0.9} if kept == "true positives" else {}
            prediction_file.write_text(json.dumps(keep))
        monkeypatch.setattr("sys.argv", ["evaluate.py", str(evaluation_fold)])

        evaluate.main()

        assert (evaluation_fold / "Evaluation@0.1.png").is_file()
