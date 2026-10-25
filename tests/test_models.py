import json

import torch

from deepanepose.utils import get_model, load_training_config


def test_default_scales_give_one_grid_matching_the_model_output(tmp_path):
    config = {
        "patch_shape": [32, 32, 32],
        "model": "YOLO_Orientation",
        "num parameters": 6,
        "in_channels": 1,
        "n_base_filters": 4,
        "layer_order": "cbl",
        "depth": 3,
        "basic_module": "ExtResNetBlock",
        "pool_type": "conv",
    }
    (tmp_path / "ndl_config.json").write_text(json.dumps(config))

    config = load_training_config(tmp_path)
    outputs = get_model(config)(torch.zeros(1, 1, 32, 32, 32))

    assert config["scales"] == [8]
    assert [output[0].shape[1:4] for output in outputs] == [(8, 8, 8)]
