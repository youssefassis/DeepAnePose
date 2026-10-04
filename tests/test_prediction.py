import numpy as np
import pytest
import torch

import prediction


class OneDetectionPerPatch(torch.nn.Module):
    """Detects an aneurysm in the first cell of a 2x2x2 grid of every patch, its neck 1 voxel away along x."""

    def forward(self, x):
        confidence = torch.full((x.shape[0], 2, 2, 2, 1), -10.0)
        confidence[:, 0, 0, 0, 0] = 10.0
        neck_offset = torch.zeros((x.shape[0], 2, 2, 2, 3))
        neck_offset[..., 0] = 0.25  # 1 voxel in 4-voxel cells
        return [[confidence, torch.zeros((x.shape[0], 2, 2, 2, 3)), neck_offset]]


@pytest.mark.parametrize("batch_size", [1, 2, 3])
def test_patch_wise_prediction_does_not_depend_on_batch_size(batch_size):
    volume = np.zeros((16, 8, 8), dtype=np.float32)  # two 8-voxel patches along x

    _, spheres = prediction.ndl_patch_wise_prediction(
        "cpu", OneDetectionPerPatch(), volume, [8, 8, 8], iou_threshold=0.01, detections_per_patch=None, batch_size=batch_size
    )

    # center: middle of the first 4-voxel cell of each patch; neck (P1): 1 voxel before along x
    assert sorted(s[:6] for s in spheres) == [[2.0, 2.0, 2.0, 1.0, 2.0, 2.0], [10.0, 2.0, 2.0, 9.0, 2.0, 2.0]]
