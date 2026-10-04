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
